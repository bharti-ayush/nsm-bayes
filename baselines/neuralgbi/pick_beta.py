import torch
from pathlib import Path
from gbi.GBI import GBInference
import numpy as np
from scipy.stats import chi2
from torch import Tensor
from experiment_helper import infer_GBI
from omegaconf import OmegaConf
from typing import List
from experiment_helper import GeneralTask

def compute_theta_hat(
    x_obs: torch.Tensor,                       # (n, d_x)
    network: GBInference,                            # fn (x_batch, theta_batch) -> log prob
    theta_init: torch.Tensor,                  # (d_theta,), starting point
    num_steps: int = 200,
    lr: float = 1e-2,
    verbose: bool = False,
) -> np.ndarray:

    theta = theta_init.clone().detach().to(device=x_obs.device, dtype=x_obs.dtype)
    theta.requires_grad_(True)

    print(theta.shape)
    print(x_obs.shape)
    optimizer = torch.optim.Adam([theta], lr=lr)
    
    if x_obs.dim() == 2:
        x_obs_reshaped = x_obs.unsqueeze(0)  # (n, d_x) -> (1, n, d_x)
    else:
        x_obs_reshaped = x_obs
    
    for step in range(num_steps):
        optimizer.zero_grad()
        # theta is a single vector, repeat it to match batch size of x_obs_reshaped
        # x_obs_reshaped is (1, n, d_x), so we need theta to be (1, d_theta)
        loss = network.distance_net(theta.unsqueeze(0), x_obs_reshaped).squeeze(1).mean()
        loss.backward()
        optimizer.step()
        
        if verbose and (step % 20 == 0 or step == num_steps - 1):
            print(f"[theta_hat optimisation] step {step:4d} | loss = {loss.item():.4f}")
    
    print("Estimated theta true: ", theta)
    return theta.detach().cpu().numpy()

def calculate_coverage(theta_samples: np.ndarray, theta_hat: np.ndarray, alpha: float) -> float:
    mu = np.mean(theta_samples, axis=0)
    Sigma = np.cov(theta_samples.T)
    Sigma_inv = np.linalg.inv(Sigma)
    thresh = chi2.ppf(1.0 - alpha, df=theta_samples.shape[1])
    d2_hat = (theta_hat - mu)[None, :] @ Sigma_inv @ (theta_hat - mu)[:, None]
    
    return int(d2_hat[0, 0] <= thresh)\

def calculate_empirical_coverage_bootstrap(inference: GBInference, 
                                 x_obs: Tensor, 
                                 theta_hat: Tensor, 
                                 beta: float, 
                                 alpha: float, 
                                 num_bootstraps: int,
                                 task, 
                                 cfg):
    emprical_coverage = 0

    for _ in range(num_bootstraps):
        # Sample with replacement: generate random indices that can repeat
        indices = torch.randint(0, x_obs.shape[0], (x_obs.shape[0],), device=x_obs.device)
        x_boot = x_obs[indices]
        theta_samples = infer_GBI(inference, x_boot, task.prior, beta, cfg.num_posterior_samples, cfg)
        theta_samples = theta_samples.numpy()
        coverage = calculate_coverage(theta_samples, theta_hat, alpha)
        emprical_coverage += coverage

    return emprical_coverage / num_bootstraps


def return_best_beta(inference: GBInference, x_obs: Tensor, beta_list: List[float], theta_init: Tensor, alpha: float, num_bootstraps: int, task: GeneralTask, cfg: OmegaConf):
    """
    Find the best beta by selecting the one with coverage closest to 1 - alpha.
    
    Args:
        inference: GBInference object
        x_obs: Observed data tensor
        beta_list: List of beta values to try
        theta_init: Initial theta value for optimization
        alpha: Significance level
        num_bootstraps: Number of bootstrap samples
        task: GeneralTask object
        cfg: Configuration object
        
    Returns:
        Tuple of (best_beta, best_coverage, beta_values, coverage_values)
    """
    theta_hat = compute_theta_hat(x_obs, inference, theta_init)

    target_coverage = 1.0 - alpha
    best_beta = None
    best_coverage = None
    best_distance = float('inf')
    beta_values = []
    coverage_values = []
    for beta in beta_list:
        emprical_coverage = calculate_empirical_coverage_bootstrap(inference, x_obs, theta_hat, beta, alpha, num_bootstraps, task, cfg)
        beta_values.append(beta)
        coverage_values.append(emprical_coverage)
        
        # Calculate distance from target coverage
        distance = abs(emprical_coverage - target_coverage)
        print(f"  Beta={beta}: Coverage={emprical_coverage:.4f} (target: {target_coverage:.4f}, distance: {distance:.4f})")
        
        if distance < best_distance:
            best_distance = distance
            best_coverage = emprical_coverage
            best_beta = beta

    return best_beta, best_coverage, beta_values, coverage_values

def test():
    import pickle
    import time
    from exp import GeneralTask

    project_root = Path(__file__).resolve().parent.parent.parent
    config_name = "gnk"
    
    inference_path = project_root / "baselines" / "results" / config_name / "neuralgbi" / "inference_run_0.pkl"
    with open(inference_path, "rb") as f:
        inference = pickle.load(f)

    data_path = project_root / "data" / config_name / "x_obs_mis_0.pkl"
    with open(data_path, "rb") as f:
        x_obs = pickle.load(f)
    theta_init = torch.tensor([0.0, 0.0, 0.0, 0.0])
    theta_hat = compute_theta_hat(x_obs, inference, theta_init)

    with open(project_root / "baselines" / "results" / config_name / "neuralgbi" / "beta_100" / "post_samples_run_0.pkl", "rb") as f:
        theta_samples = pickle.load(f)
        
    theta_samples = theta_samples.numpy()
    coverage = calculate_coverage(theta_samples, theta_hat, 0.05)
    print(f"Coverage: {coverage}")


    # Calculate empirical coverage using bootstrap
    cfg = OmegaConf.load(project_root / "rca_sbi" / "config" / "gnk.yaml")

    dtype = torch.float32
    prior_mean = torch.tensor(cfg.prior_mean, dtype = dtype)
    prior_cov = torch.tensor(cfg.prior_cov, dtype = dtype)    
    d_x = cfg.d_x
    d_theta = len(prior_mean)
    task = GeneralTask(prior_mean, prior_cov, d_theta, d_x)
    t0 = time.time()

    emprical_coverage = calculate_empirical_coverage_bootstrap(inference, x_obs, theta_init, 
                                                               beta = 100, alpha = 0.05, 
                                                               num_bootstraps = 100, task = task, cfg = cfg)
                                                            
    t1 = time.time()
    print(f"Time taken: {t1 - t0} seconds")
    print(f"Empirical coverage: {emprical_coverage}")


if __name__ == "__main__":
    test()