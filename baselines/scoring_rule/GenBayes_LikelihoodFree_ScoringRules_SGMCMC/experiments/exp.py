"""
Standard g-and-k experiment matching other baselines format.
Usage: python exp_gnk_standard.py [experiment_name] --index [run_index]
Example: python exp_gnk_standard.py gnk --index 0
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import pickle
import functools
import jax
import gc
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))  # Add project root so rca_sbi can be imported
sys.path.insert(0, str(project_root / "baselines" / "scoring_rule" / "GenBayes_LikelihoodFree_ScoringRules_SGMCMC"))

from src.scoring_rules.scoring_rules import KernelScore
from src.transformers import BoundedVarTransformer
from src.sampler.sgMCMC import SGMCMC
from experiment_helper import (
    estimate_bandwidth_from_data, 
    torch_uni_g_and_k, 
    torch_sir,
    joint_log_prob,
    return_best_beta,
    find_theta_minimizing_kernel_score
)

import time

test_mode = False
torch.set_default_dtype(torch.float64) # because of this, file size is double of other method (other method is float32)


def main(experiment_name, run_idx=None): 
    # Load config
    config_path = project_root / "rca_sbi" / "config" / f"{experiment_name}.yaml"
    cfg = OmegaConf.load(config_path)
    
    # Extract config values
    num_repeat = cfg.num_repeat
    # num_posterior_samples = cfg.num_posterior_samples + cfg.warmup_steps
    
    # If run_idx is not provided, validate it's within range
    if run_idx is None:
        raise ValueError("--index argument is required. Please specify which run to execute (0-indexed).")
    
    if run_idx < 0 or run_idx >= num_repeat:
        raise ValueError(f"run_idx {run_idx} is out of range. Must be between 0 and {num_repeat-1}.")

    if experiment_name in ["gnk", "turin"]:
        prior_mean = np.array(cfg.prior_mean, dtype=np.float64)
        prior_cov = np.array(cfg.prior_cov, dtype=np.float64)
    elif experiment_name in ["sir_undercounting", "sir_student_t_2", "sir_student_t_1"]:
        prior_mean = torch.tensor([
            torch.log(torch.tensor(cfg.prior_mean[0])),         # log beta
            torch.log(torch.tensor(cfg.prior_mean[1])),        # log gamma
            torch.logit(torch.tensor(cfg.prior_mean[2])),        # logit rho
            torch.log(torch.tensor(cfg.prior_mean[3])),           # log I0
        ],  dtype = torch.float64).numpy()
        prior_std  = torch.tensor(cfg.prior_std, dtype=torch.float64) 
        prior_cov = torch.diag(prior_std ** 2).numpy()
    
    # Setup paths
    save_dir = project_root / "baselines" / "results" / experiment_name / "scoring_rule_sgmcmc"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Parameter bounds - all unbounded
    L_model = np.array([None, None, None, None])  # All unbounded
    U_model = np.array([None, None, None, None])   # All unbounded
    b = BoundedVarTransformer(lower_bound=L_model, upper_bound=U_model)
    
    # Model
    if experiment_name == "gnk":
        model = torch_uni_g_and_k()

    elif experiment_name in ["sir_undercounting", "sir_student_t_2", "sir_student_t_1"]:
        model = torch_sir(T_sir=cfg.T_sir, N_sir=cfg.N_sir)

    else:
        raise ValueError(f"Invalid experiment name: {experiment_name}")

    data_dir = project_root / "data" / experiment_name

    # Setup joint log prob function (will be updated per run with correct scoring rule)    
    beta_values = [1, 10, 100] if test_mode is False else [1]

    bootstrap_size = 20 # if test_mode is False else 2
    num_posterior_bootstrap = cfg.sr_num_posterior_bootstrap
    num_sample_per_param = cfg.sr_num_samples_per_param
    num_sample_per_param_bootstrap = cfg.sr_num_samples_per_param_boot
    grid_size_beta = len(beta_values)
    num_posterior = 500
    warmup = 500
    num_posterior_samples = num_posterior + warmup

    # Budget calculation: theta_hat computation + calibration + final posterior sampling
    # For theta_hat optimization: n_steps * n_sim_per_theta
    n_steps_theta_hat = 200  # Number of optimization steps for theta_hat
    budget_theta_hat = n_steps_theta_hat * num_sample_per_param_bootstrap  # Optimization budget for theta_hat
    budget_picking_beta = grid_size_beta * bootstrap_size * num_posterior_bootstrap * num_sample_per_param_bootstrap
    budget_posterior_sampling = (num_posterior + warmup) * num_sample_per_param  # One beta (best_beta)
    budget = budget_theta_hat + budget_picking_beta + budget_posterior_sampling
    print(f"budget_theta_hat: {budget_theta_hat}")
    print(f"budget_picking_beta: {budget_picking_beta}")
    print(f"budget_posterior_sampling: {budget_posterior_sampling}")
    print(f"budget: {budget}")

    print(f"beta_values: {beta_values}")
    print(f"bootstrap_size: {bootstrap_size}")
    print(f"num_posterior_bootstrap: {num_posterior_bootstrap}")
    print(f"num_sample_per_param: {num_sample_per_param}")
    print(f"num_sample_per_param_bootstrap: {num_sample_per_param_bootstrap}")
    print(f"num_posterior_samples: {num_posterior_samples}")
    
    # Run for the specified index
    print(f"\n{'='*60}")
    print(f"Run {run_idx+1}/{num_repeat}")
    print(f"{'='*60}")
    
    # Set random seeds
    np.random.seed(run_idx + 1)
    torch.manual_seed(run_idx + 1)
    jax.random.PRNGKey(run_idx + 1)
    
    # Load observed data
    observed_data_path = data_dir / f"x_obs_mis_{run_idx}.pkl"
    with open(observed_data_path, "rb") as f:
        obs = pickle.load(f).float()
    
    # Estimate bandwidth and create scoring rule for this run
    sigma = estimate_bandwidth_from_data(obs.numpy())

    time_total = 0

    # Step 1: Compute theta_hat using optimization (minimizing kernel score)
    print(f"\n{'='*60}")
    print(f"Computing theta_hat for run {run_idx+1}/{num_repeat}")
    print(f"{'='*60}")
    
    t0 = time.time()
    theta_hat_torch = find_theta_minimizing_kernel_score(
        model=model,
        x_obs=obs,
        n_sim_per_theta=num_sample_per_param_bootstrap,
        n_steps=n_steps_theta_hat,
        lr=1e-2,
        theta_init=prior_mean
    )
    t1 = time.time()
    time_thata_hat = t1 - t0
    time_total += time_thata_hat
    
    theta_hat = theta_hat_torch.detach().cpu().numpy()
    print(f"theta_hat computed: {theta_hat}")
    
    # Clean up
    del theta_hat_torch
    model.scores.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 2: Calibrate beta to determine best value
    print(f"\n{'='*60}")
    print(f"Calibrating beta for run {run_idx+1}/{num_repeat}")
    print(f"{'='*60}")
    
    # Run beta calibration
    t2 = time.time()
    best_beta, best_coverage, beta_values_cal, coverage_values, optimal_dt_dict = return_best_beta(
        model=model,
        x_obs=obs,
        beta_list=beta_values,
        theta_hat=theta_hat,
        alpha=cfg.alpha,
        num_bootstraps=bootstrap_size,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        n_samples_per_param=num_sample_per_param_bootstrap,
        num_posterior_samples=num_posterior_bootstrap, # no warmup
        transformer=b,
        run_idx=run_idx
    )
    t3 = time.time()
    time_beta_calibration = t3 - t2
    time_total += time_beta_calibration
    print(f"Best beta: {best_beta}, Best coverage: {best_coverage:.4f}")
    
    # Step 3: Run full posterior sampling only with best_beta
    print(f"\n{'='*60}")
    print(f"Running posterior sampling with best beta={best_beta} for run {run_idx+1}/{num_repeat}")
    print(f"{'='*60}")
    
    save_dir_beta = save_dir / f"beta_{best_beta}"
    save_dir_beta.mkdir(parents=True, exist_ok=True)

    ks = KernelScore(weight=best_beta, sigma=sigma)
    joint_log_prob_func = functools.partial(
        joint_log_prob, 
        model=model, 
        scoring_rule=ks,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        n_samples_per_param=num_sample_per_param
    )
    
    sampler = SGMCMC(
        model, 
        observations=obs, 
        joint_log_prob=joint_log_prob_func, 
        transformer=b, 
        n_samples=num_posterior_samples,
        seed=run_idx
    )
    
    init_params = jax.numpy.zeros(model.param_dim)
    
    # Use optimal_dt from calibration if available
    optimal_dt = optimal_dt_dict.get(best_beta)
    t4 = time.time()
    if optimal_dt is not None:
        op, _ = sampler.sample(
            init_params=init_params,
            use_optim=False, 
            use_mamba=False,
            step_size=optimal_dt
        )
    else:
        op, optimal_dt = sampler.sample(
            init_params=init_params,
            use_optim=False, 
            use_mamba=True,
        )
    t5 = time.time()
    time_posterior_sampling = t5 - t4
    time_total += time_posterior_sampling

    # Extract posterior samples from output
    samples_uncon = op['samples_uncon']
    post_samples = samples_uncon.numpy()
    post_samples = post_samples[warmup:,:]  # remove burn-in samples
    
    # Save posterior samples for best beta
    output_path = save_dir_beta / f"post_samples_run_{run_idx}.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(post_samples, f)

    print(f"Results saved to {output_path}")
    
    # Save optimal_dt_dict
    with open(save_dir_beta / f"optimal_dt_run_{run_idx}.pkl", "wb") as f:
        pickle.dump(optimal_dt_dict, f)
    
    # Save best_beta as text file
    with open(save_dir / f"best_beta_run_{run_idx}.txt", "w") as f:
        f.write(str(best_beta))
    
    # Save beta_values and coverage_values
    with open(save_dir / f"beta_coverage_values_run_{run_idx}.txt", "w") as f:
        # First row: beta_values
        f.write(" ".join(str(b) for b in beta_values_cal) + "\n")
        # Second row: coverage_values
        f.write(" ".join(str(c) for c in coverage_values) + "\n")


    # Copy posterior samples from best beta to main save directory
    best_beta_samples_path = save_dir / f"beta_{best_beta}" / f"post_samples_run_{run_idx}.pkl"
    if best_beta_samples_path.exists():
        with open(best_beta_samples_path, "rb") as f:
            best_post_samples = pickle.load(f)
        output_path = save_dir / f"post_samples_run_{run_idx}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(best_post_samples, f)
        print(f"Best beta posterior samples saved to {output_path}")
        del best_post_samples
    
    # Save total time as pkl
    with open(save_dir / f"time_total_run_{run_idx}.pkl", 'wb') as f:
        pickle.dump(time_total, f)
    with open(save_dir / f"time_theta_hat_run_{run_idx}.pkl", 'wb') as f:
        pickle.dump(time_thata_hat, f)
    with open(save_dir / f"time_beta_calibration_run_{run_idx}.pkl", 'wb') as f:
        pickle.dump(time_beta_calibration, f)
    with open(save_dir / f"time_posterior_sampling_run_{run_idx}.pkl", 'wb') as f:
        pickle.dump(time_posterior_sampling, f)


    
    # Clean up memory after each run
    try:
        del obs, theta_hat, best_beta, best_coverage, beta_values_cal, coverage_values
    except NameError:
        pass  # Some variables might not exist
    model.scores.clear()  # Clear accumulated scores
    gc.collect()  # Force garbage collection    
    # Clear PyTorch cache if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print(f"\n{'='*60}")
    print(f"Run {run_idx+1}/{num_repeat} completed!")
    print(f"Results saved to: {save_dir}")
    print(f"Posterior time: {time_total:.2f} seconds")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment for a specific run index")
    parser.add_argument("experiment_name", type=str, help="Name of the experiment (e.g., 'gnk', 'sir_undercounting')")
    parser.add_argument("--index", type=int, required=True, help="Run index (0-indexed)")
    
    args = parser.parse_args()
    
    main(experiment_name=args.experiment_name, run_idx=args.index)
    print(f"Experiment name: {args.experiment_name}, Run index: {args.index}")