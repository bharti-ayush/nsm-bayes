"""
Experiment script for g-and-k simulator using NeuralGBI (GBI method).

This script follows the neuralgbi pattern but adapted for the baseline framework.
It trains GBI and runs inference for multiple runs, saving results in the same
format as other baselines.
"""

import torch
import numpy as np
import time
import pickle
import sys
import random
from pathlib import Path
from torch.distributions import MultivariateNormal

# Add paths
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from gbi.distances import mmd_dist
from omegaconf import OmegaConf
from experiment_helper import train_GBI, infer_GBI, compute_mmd_lengthscale, GeneralTask
from pick_beta import return_best_beta
import sys

import torch
from functools import partial

def main(config_name, beta = 100.0, pretrained = False, posterior_obtained = False, beta_tuning = False):
    """Main function that loads config and runs experiment.
    
    Args:
        config_name: Name of the config file (without .yaml extension)
    """
    # Load config manually
    config_path = Path(__file__).resolve().parent.parent.parent / "rca_sbi" / "config"
    cfg = OmegaConf.load(config_path / f"{config_name}.yaml")
    
    # Setup paths
    project_root = Path(__file__).resolve().parent.parent.parent
    save_dir_inference = project_root / "baselines" / "results" / config_name / "neuralgbi_more_budget" # save inference object here
    save_dir = project_root / "baselines" / "results" / config_name / "neuralgbi_more_budget" /  f"beta_{int(beta)}"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_dir_inference.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32
    
    # Load configs from rca_sbi/config/sir.yaml
    if config_name in ["sir", "sir_undercounting", "sir_student_t_1", "sir_student_t_2"]:
        prior_mean_before_transform = torch.tensor(cfg.prior_mean, dtype = dtype)
        prior_mean = torch.tensor([
        torch.log(torch.tensor(prior_mean_before_transform[0])),         # log beta
        torch.log(torch.tensor(prior_mean_before_transform[1])),        # log gamma
        torch.logit(torch.tensor(prior_mean_before_transform[2])),        # logit rho
        torch.log(torch.tensor(prior_mean_before_transform[3])),           # log I0
        ])
        prior_std = torch.tensor(cfg.prior_std, dtype = dtype)
        prior_cov = torch.diag(prior_std ** 2)
        training_data_dir = project_root / "data" / "sir" # training data is same all the time.
        data_dir = project_root / "data" / config_name


    elif config_name in ["gnk", "turin"]:
        prior_mean = torch.tensor(cfg.prior_mean, dtype = dtype)
        prior_cov = torch.tensor(cfg.prior_cov, dtype = dtype)
        data_dir = project_root / "data" / config_name
        training_data_dir = data_dir
    
    d_x = cfg.d_x
    d_theta = len(prior_mean)
    task = GeneralTask(prior_mean, prior_cov, d_theta, d_x)
    num_posterior_samples = cfg.num_posterior_samples 
    num_repeat = cfg.num_repeat

    #  Distance function and beta
    ## Use mmd_dist for multiple iid observations per theta (like gaussian_mixture)

    # GBI hyperparameters (default values from benchmark/run_algorithms/config/algorithm/GBI.yaml)
    gbi_config = {
        'net_type': 'resnet',
        'training_batch_size': 500,
        'num_layers': 3,
        'num_hidden': 64,
        'positive_constraint_fn': 'softplus',
        'max_epochs': 5000,
        'n_epochs_convergence': 100,
        'print_every_n': 20,
        'validation_fraction': 0.1,
        'noise_level': 2.0,
        'n_augmented_x': 100,
        'do_precompute_distances': False,
        'n_train_per_theta': 2,
        'n_val_per_theta': 5,
        'train_with_obs': True,
    }
    
    train_times = []
    posterior_times = []
    beta_tuning_times = []

    for ind in range(num_repeat):
    
        random.seed(ind)
        np.random.seed(ind)
        torch.manual_seed(ind)

        # Load observed data
        path_x_obs = data_dir / f"x_obs_mis_{ind}.pkl"

        print(f"--------------------------------Run {ind}--------------------------------")
        print("Loading observed data from: ", path_x_obs)
        with open(path_x_obs, "rb") as f:
            x_obs = pickle.load(f)


        if pretrained:
            print("Loading pretrained inference from: ", save_dir_inference / f"inference_run_{ind}.pkl")
            with open(save_dir_inference / f"inference_run_{ind}.pkl", "rb") as f:
                inference = pickle.load(f)
        else:
            # Load training data
            path_theta = training_data_dir / f"theta_{ind}_ace_more_budget.pt"
            path_x_sim = training_data_dir / f"x_sim_{ind}_ace_more_budget.pt"
            print("Loading training data from: ", path_theta, path_x_sim)
            theta = torch.load(path_theta).float()
            x_sim = torch.load(path_x_sim).float()
            print(theta.shape, x_sim.shape)
            # Train GBI
            t0 = time.time()
            print("Calculating MMD lengthscale...")
            lengthscale = compute_mmd_lengthscale(x_obs)
            print(f"Lengthscale: {lengthscale}")
            distance_func = partial(mmd_dist, lengthscale=lengthscale)
            print("Training GBI...")
            inference = train_GBI(theta, x_sim, x_obs, task, distance_func, gbi_config)
            t1 = time.time()

            with open(save_dir_inference / f"inference_run_{ind}.pkl", "wb") as f:
                pickle.dump(inference, f)
            
            train_time = t1 - t0
            train_times.append(train_time)
        
        # Run inference
        if not posterior_obtained:
            print("Sampling from the posterior...")
            t2 = time.time()
            post_samples = infer_GBI(
                inference, x_obs, task.prior, beta, num_posterior_samples, cfg,
            )
            t3 = time.time()
            posterior_time = t3 - t2
            posterior_times.append(posterior_time)        
            # Save results
            with open(save_dir / f"post_samples_run_{ind}.pkl", "wb") as f:
                pickle.dump(post_samples, f)
            
            print(f"Results saved to {save_dir / f'post_samples_run_{ind}.pkl'}")

        
        # Pick best beta
        if beta_tuning:
            """
            Only run after obtaining all the posterior samples.
            beta argument is used 
            """
            print("Tuning beta...")
            beta_list = [1, 10, 100, 1000, 10000] # ascending order!
            posterior_samples_base = pickle.load(open(save_dir / f"post_samples_run_{ind}.pkl", "rb")) # base posterior samples to calculate initial value (the one with beta argument)

            t4 = time.time()
            theta_init = posterior_samples_base.mean(dim=0) # initial value for the optimisation
            best_beta, best_coverage, beta_values, coverage_values = return_best_beta(inference, x_obs, beta_list, theta_init, alpha = cfg.alpha, num_bootstraps = 20, task = task, cfg = cfg)
            t5 = time.time()

            beta_tuning_time = t5 - t4
            print("Best beta: ", best_beta, "Best coverage: ", best_coverage)
            # Save best_beta as text file
            with open(save_dir_inference / f"best_beta_run_{ind}.txt", "w") as f:
                f.write(str(best_beta))
            # Save beta_values and coverage_values in a text file (beta_values in first row, coverage_values in second row)
            with open(save_dir_inference / f"beta_coverage_values_run_{ind}.txt", "w") as f:
                # First row: beta_values
                f.write(" ".join(str(b) for b in beta_values) + "\n")
                # Second row: coverage_values
                f.write(" ".join(str(c) for c in coverage_values) + "\n")

            beta_tuning_times.append(beta_tuning_time)


    # Save timing information
    if not pretrained:
        with open(save_dir_inference / "train_times.pkl", "wb") as f:
            pickle.dump(train_times, f)

    if not posterior_obtained:
        with open(save_dir / "posterior_times.pkl", "wb") as f:
            pickle.dump(posterior_times, f)

    if beta_tuning:
        with open(save_dir_inference / "beta_tuning_times.pkl", "wb") as f:
            pickle.dump(beta_tuning_times, f)


if __name__ == "__main__":
    config_name = sys.argv[1]
    pretrained = sys.argv[2].lower() == 'true' if len(sys.argv) > 2 else False # if True, load pretrained inference object
    posterior_obtained = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False # if True, load posterior samples
    beta_tuning = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else False # if True, run tunning for beta
    beta = int(sys.argv[5]) if len(sys.argv) > 5 else 100

    main(config_name=config_name, beta = beta, pretrained = pretrained, posterior_obtained = posterior_obtained, beta_tuning = beta_tuning)

