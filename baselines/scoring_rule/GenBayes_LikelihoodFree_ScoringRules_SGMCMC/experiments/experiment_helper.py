"""
Helper functions for scoring rule experiments.
Contains bandwidth estimation, simulator classes, and calibration functions.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import functools
import jax
import gc
from scipy.spatial.distance import cdist
from scipy.stats import chi2
from typing import List

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))  # Add project root so rca_sbi can be imported
sys.path.insert(0, str(project_root / "baselines" / "scoring_rule" / "GenBayes_LikelihoodFree_ScoringRules_SGMCMC"))

from rca_sbi.simulators import simulate_sir, sir_summary, TurinModel
from src.scoring_rules.scoring_rules import KernelScore
from src.sampler.sgMCMC import SGMCMC
from torch.distributions import MultivariateNormal


def estimate_bandwidth_from_data(observations):
    """Estimate the bandwidth for the gaussian kernel from observed data.
    
    This uses the same median heuristic as Robust-SBI: sqrt(median(squared_distances / 2))
    
    Args:
        observations: Array of shape (n_obs, d_x) containing observed data
        
    Returns:
        Bandwidth estimate: sqrt(median(squared_distances / 2))
    """    
    # Compute pairwise squared Euclidean distances
    squared_distances = cdist(observations, observations, 'sqeuclidean')
    
    # Includes diagonal (which are 0) in median calculation
    return np.sqrt(np.median(squared_distances / 2))


class torch_uni_g_and_k(torch.nn.Module):
    """
    G-and-k distribution simulator.
    Takes as parameters (A, log(B), g, log(k)) in unconstrained space.
    Outputs a draw from the g-and-k distribution.
    """

    def __init__(self):
        super().__init__()
        self.c = 0.8
        self.param_dim = 4
        self.scores = []  # for debugging purposes, just to save the score

    def torch_forward_simulate(self, params, num_forward_simulations: int, seed=None):
        """
        Simulate from g-and-k distribution.
        
        Args:
            params: tensor of shape (4,) containing [A, log(B), g, log(k)]
            num_forward_simulations: int
            seed: Optional random seed
            
        Returns:
            Simulated data of shape (num_forward_simulations, 1)
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        A, logB, g, logk = params[0], params[1], params[2], params[3]
        # Transform from log space to model space
        B = torch.exp(logB)
        k = torch.exp(logk)
        z = torch.randn(num_forward_simulations)
        
        # Compute g-and-k transformation
        term_g = torch.tanh(0.5 * g * z)
        term_k = (1 + z**2)**k
        result = A + B * (1 + 0.8 * term_g) * term_k * z
        
        return result.reshape(num_forward_simulations, 1)


class torch_sir(torch.nn.Module):
    """
    SIR (Susceptible-Infected-Recovered) model simulator.
    Takes as parameters (log(beta), log(gamma), logit(rho), log(I0)) in unconstrained space.
    Outputs summary statistics from SIR simulation.
    """

    def __init__(self, T_sir, N_sir):
        super().__init__()
        self.param_dim = 4
        self.d_x = 3
        self.scores = []  # for debugging purposes, just to save the score
        self.T_sir = T_sir
        self.N_sir = N_sir

    def torch_forward_simulate(self, params, num_forward_simulations: int, seed=None):
        """
        Simulate from SIR model.
        
        Args:
            params: tensor of shape (4,) containing [log(beta), log(gamma), logit(rho), log(I0)]
            num_forward_simulations: int
            seed: Optional random seed
            
        Returns:
            Summary statistics of shape (num_forward_simulations, d_x)
        """
        params = params.unsqueeze(0).repeat(num_forward_simulations, 1)

        x_raw = simulate_sir(params, T=self.T_sir, N=self.N_sir)
        x = sir_summary(x_raw, self.N_sir)

        return x.reshape(num_forward_simulations, self.d_x)


class torch_turin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param_dim = 4
        self.B = 4e9
        self.Ns = 801
        self.tau0 = 0,
        self.output = "moments"
        self.epsilon = 0.0,
        self.device = "cpu"
        self.d_x = 3
        self.scores = []  # for debugging purposes, just to save the score

    def torch_forward_simulate(self, params, num_forward_simulations: int, seed=None):

        params = params.unsqueeze(0).repeat(num_forward_simulations, 1)

        print(params.shape)

        x = TurinModel(params, B=self.B, Ns=self.Ns, N=num_forward_simulations, tau0=self.tau0, output=self.output, epsilon=self.epsilon, device=self.device)

        print(x.shape)
        return x.reshape(num_forward_simulations, self.d_x)


def joint_log_prob(param_torch_unconstrained, obs, model, scoring_rule, prior_mean, prior_cov, n_samples_per_param=500):
    """
    Joint log probability function for MCMC sampling.
    For unbounded parameters, transformation is identity.
    """
    # For unbounded parameters, no transformation needed (identity)
    param_torch_constrained = param_torch_unconstrained
    laj = torch.tensor(0.0, dtype=torch.float64)  # Identity transformation has zero log Jacobian
    
    # Calculate log prior (Gaussian prior from config)
    prior_dist = MultivariateNormal(
        loc=torch.tensor(prior_mean, dtype=torch.float64),
        covariance_matrix=torch.tensor(prior_cov, dtype=torch.float64)
    )
    log_prior = prior_dist.log_prob(param_torch_constrained)
    
    # Calculate log likelihood / score
    sims = model.torch_forward_simulate(param_torch_constrained, n_samples_per_param)
    log_ll = scoring_rule.loglikelihood(y_obs=obs, y_sim=sims, use_torch=True)
    model.scores.append(log_ll.detach())  # Detach otherwise mem explodes!
    
    return laj + log_prior + log_ll


def calculate_coverage(theta_samples: np.ndarray, theta_hat: np.ndarray, alpha: float) -> float:
    """
    Calculate coverage: whether theta_hat is within the (1-alpha) confidence region.
    
    Args:
        theta_samples: Posterior samples of shape (n_samples, d_theta)
        theta_hat: True/estimated theta of shape (d_theta,)
        alpha: Significance level
        
    Returns:
        1 if theta_hat is covered, 0 otherwise
    """
    mu = np.mean(theta_samples, axis=0)
    Sigma = np.cov(theta_samples.T)
    Sigma_inv = np.linalg.inv(Sigma)
    thresh = chi2.ppf(1.0 - alpha, df=theta_samples.shape[1])
    d2_hat = (theta_hat - mu)[None, :] @ Sigma_inv @ (theta_hat - mu)[:, None]
    
    return int(d2_hat[0, 0] <= thresh)


def calculate_empirical_coverage_bootstrap(
    model,
    x_obs: torch.Tensor,
    theta_hat: np.ndarray,
    beta: float,
    alpha: float,
    num_bootstraps: int,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    n_samples_per_param: int,
    num_posterior_samples: int,
    transformer,
    run_idx: int,
) -> float:
    """
    Calculate empirical coverage using bootstrap resampling.
    
    Args:
        model: Model with torch_forward_simulate method
        x_obs: Observed data tensor
        theta_hat: True/estimated theta
        beta: Beta value for scoring rule
        alpha: Significance level
        num_bootstraps: Number of bootstrap samples
        scoring_rule_base: Base scoring rule (will be updated with beta)
        prior_mean: Prior mean
        prior_cov: Prior covariance
        n_samples_per_param: Number of simulations per parameter
        num_posterior_samples: Number of posterior samples to generate
        warmup_steps: Number of warmup steps to discard
        transformer: Parameter transformer
        run_idx: Run index for random seed
        
    Returns:
        Average empirical coverage across bootstrap samples
    """
    empirical_coverage = 0.0
    
    for bootstrap_idx in range(num_bootstraps):
        # Sample with replacement: generate random indices that can repeat
        torch.manual_seed(run_idx * 1000 + bootstrap_idx)
        indices = torch.randint(0, x_obs.shape[0], (x_obs.shape[0],), device=x_obs.device)
        x_boot = x_obs[indices]
        
        # Create scoring rule with current beta
        sigma = estimate_bandwidth_from_data(x_boot.numpy())
        ks = KernelScore(weight=beta, sigma=sigma)
        
        # Create joint log prob function
        joint_log_prob_func = functools.partial(
            joint_log_prob,
            model=model,
            scoring_rule=ks,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            n_samples_per_param=n_samples_per_param
        )
        
        # Initialize sampler
        sampler = SGMCMC(
            model,
            observations=x_boot,
            joint_log_prob=joint_log_prob_func,
            transformer=transformer,
            n_samples=num_posterior_samples,
            seed=run_idx * 1000 + bootstrap_idx  # Different seed for each bootstrap
        )
        
        # Initialize parameters to zero
        init_params = jax.numpy.zeros(model.param_dim)
        
        # Run sampler
        if bootstrap_idx == 0:
            op, optimal_dt = sampler.sample(
                init_params=init_params,
                use_optim=False,
                use_mamba=True,
            )
        else:
            # reuse optimal_dt when beta is the same
            op, _  = sampler.sample(
                init_params=init_params,
                use_optim=False,
                use_mamba=False,
                step_size = optimal_dt
            )
        
        # Extract posterior samples
        samples_uncon = op['samples_uncon']
        theta_samples = samples_uncon.numpy()        
        # Calculate coverage
        coverage = calculate_coverage(theta_samples, theta_hat, alpha)
        empirical_coverage += coverage
    
    return empirical_coverage / num_bootstraps, optimal_dt


def return_best_beta(
    model,
    x_obs: torch.Tensor,
    beta_list: List[float],
    theta_hat: np.ndarray,
    alpha: float,
    num_bootstraps: int,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    n_samples_per_param: int,
    num_posterior_samples: int,
    transformer,
    run_idx: int) -> tuple:
    """
    Find the best beta by selecting the one with coverage closest to 1 - alpha.
    
    Args:
        model: Model with torch_forward_simulate method
        x_obs: Observed data tensor
        beta_list: List of beta values to try
        theta_hat: True/estimated theta (posterior mean from beta=1)
        alpha: Significance level
        num_bootstraps: Number of bootstrap samples
        scoring_rule_base: Base scoring rule
        prior_mean: Prior mean
        prior_cov: Prior covariance
        n_samples_per_param: Number of simulations per parameter
        num_posterior_samples: Number of posterior samples
        warmup_steps: Number of warmup steps
        transformer: Parameter transformer
        run_idx: Run index
        optimal_dt_dict: Dictionary of optimal dt values for each beta
        
    Returns:
        Tuple of (best_beta, best_coverage, beta_values, coverage_values, optimal_dt_dict)
    """
    target_coverage = 1.0 - alpha
    best_beta = None
    best_coverage = None
    best_distance = float('inf')
    beta_values = []
    coverage_values = []
    optimal_dt_dict = {beta: None for beta in beta_list}
    
    for beta in beta_list:
        print(f"  Testing beta={beta}...")
        empirical_coverage, optimal_dt_beta = calculate_empirical_coverage_bootstrap(
            model, x_obs, theta_hat, beta, alpha, num_bootstraps,
            prior_mean, prior_cov, n_samples_per_param,
            num_posterior_samples, transformer, run_idx,
        )
        beta_values.append(beta)
        coverage_values.append(empirical_coverage)
        
        # Calculate distance from target coverage
        distance = abs(empirical_coverage - target_coverage)
        print(f"    Coverage: {empirical_coverage:.4f} (target: {target_coverage:.4f}, distance: {distance:.4f})")
        
        if distance < best_distance:
            best_distance = distance
            best_coverage = empirical_coverage
            best_beta = beta

        optimal_dt_dict[beta] = optimal_dt_beta
        
        # Clean up memory after each beta test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return best_beta, best_coverage, beta_values, coverage_values, optimal_dt_dict


def find_theta_minimizing_kernel_score(model, x_obs: torch.Tensor, n_sim_per_theta: int = 500,
                                       n_steps: int = 200, lr: float = 1e-2, 
                                       theta_init: torch.Tensor = None) -> torch.Tensor:
    """
    Minimize the kernel score between observed data and model simulations:
        theta* = argmin_theta KernelScore(x_obs, model(theta)).

    Args:
        model: torch module with `param_dim` and `torch_forward_simulate(theta, n_sim, seed=None)`.
        x_obs: Observed data tensor of shape (n_obs, d_x).
        n_sim_per_theta: Number of simulations per theta when evaluating the score.
        n_steps: Number of gradient steps.
        lr: Learning rate for Adam.
        theta_init: Initial value for theta (torch tensor or numpy array). If None, uses zeros.

    Returns:
        theta_opt: Optimized theta (torch tensor of shape (param_dim,)).
    """
    x_obs = x_obs.detach().to(dtype=torch.float64)

    sigma = estimate_bandwidth_from_data(x_obs.numpy())
    ks = KernelScore(weight=1.0, sigma=sigma)

    # Initialize theta with prior mean if provided, otherwise use zeros
    if theta_init is not None:
        if isinstance(theta_init, np.ndarray):
            theta = torch.tensor(theta_init, dtype=torch.float64, requires_grad=True)
        else:
            theta = theta_init.clone().detach().to(dtype=torch.float64)
            theta.requires_grad_(True)
    else:
        theta = torch.zeros(model.param_dim, dtype=torch.float64, requires_grad=True)
    
    optimizer = torch.optim.Adam([theta], lr=lr)

    for _ in range(n_steps):
        optimizer.zero_grad()
        sims = model.torch_forward_simulate(theta, n_sim_per_theta)
        score = ks.score(observations=x_obs, simulations=sims, use_torch=True)
        score.backward()
        optimizer.step()

    return theta.detach()

