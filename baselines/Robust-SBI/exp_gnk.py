import torch
import numpy as np
from torch.distributions import MultivariateNormal
import sys
from pathlib import Path

# Add project root to path to access rca_sbi config
# exp_gnk.py is in baselines/Robust-SBI/, so go up 2 levels to get to project root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


from omegaconf import OmegaConf

from utils.get_nn_models import *
from inference.snpe.snpe_c import SNPE_C as SNPE
from inference.base import *
from utils.torchutils import *
import pickle
import os
import random
from rca_sbi.simulators import sample_gandk_fully_reparameterized
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_simulations_npe(prior, simulator, num_samples, n_obs, data_dim):
    # Sample from prior
    theta = prior.sample([num_samples])
    x_sim = torch.zeros(num_samples, n_obs, data_dim) 
    # Sample from the simulator
    for i in range(num_samples):
        for j in range(n_obs):
            x_sim[i, j, :] = simulator(theta[i, :])
    return theta, x_sim

def main(config_name: str, lambda_value: float = 1.0):
    # Use project_root instead of get_original_cwd() to avoid path issues when running from Robust-SBI directory
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # Load config manually
    config_path = project_root / "rca_sbi" / "config"
    cfg = OmegaConf.load(config_path / f"{config_name}.yaml")
    
    # experiment_name is the same as config_name
    experiment_name = config_name
    
    save_dir = project_root / "baselines" / "results" / experiment_name / ("robust_summary" + f"_lambda_{lambda_value}")
    save_dir.mkdir(parents=True, exist_ok=True)
    path_x_obs = project_root / "data" / experiment_name
    dtype = torch.float32

    # load configs from gnk.yaml
    num_repeat = cfg.num_repeat 
    prior_mean = torch.tensor(cfg.prior_mean, dtype=dtype)
    prior_cov  = torch.tensor(cfg.prior_cov,  dtype=dtype)
    num_posterior_samples = cfg.num_posterior_samples
    
    # use dafault setting for hyperparameters of robust summary paper
    distance = "mmd"
    beta = lambda_value

    n_obs = cfg.n_obs
    num_samples_nle = cfg.num_samples
    num_samples = num_samples_nle // n_obs # to fix the number of samples (n \times n_obs) fixed

    train_times = []
    posterior_times = []

    for ind in range(num_repeat):
        random.seed(ind+1)
        np.random.seed(ind+1)
        torch.manual_seed(ind+1)
        d_x = 1

        # Generate training data
        prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
        theta, x_sim = get_simulations_npe(prior, sample_gandk_fully_reparameterized, num_samples, n_obs, d_x)

        # with open(save_dir / f"theta_robust_summary_{ind}.pkl", "wb") as f:
        #     pickle.dump(theta, f)
        # with open(save_dir / f"x_sim_theta_robust_summary_{ind}.pkl", "wb") as f:
        #     pickle.dump(x_sim, f)

        x_obs = torch.tensor(np.load(path_x_obs / f"x_obs_mis_{ind}.pkl", allow_pickle=True))
        x_obs = x_obs.unsqueeze(0).unsqueeze(1)

        # Simple embedding network for g-and-k data (batch, n_obs, d_x)
        sum_net = GNKEmbedding(n_obs=n_obs, d_x=d_x, hidden_dim=20, output_dim=20).to(device)

        neural_posterior = posterior_nn(
            model = "maf",
            embedding_net = sum_net,
            hidden_features = 20,
            num_transforms = 3,
        )
        t0 = time.time()
        inference = SNPE(prior=prior, density_estimator=neural_posterior, device=str(device))
        density_estimator = inference.append_simulations(theta, x_sim.unsqueeze(1)).train(
                             distance = distance, x_obs = x_obs, beta = beta)


        t1 = time.time()
        posterior = inference.build_posterior(density_estimator, prior = prior)
        post_samples = posterior.sample((num_posterior_samples,), x = x_obs)
        t2 = time.time()
        train_time = t1 - t0
        posterior_time = t2 - t1
        train_times.append(train_time)
        posterior_times.append(posterior_time)


        with open(save_dir / f"posterior_run_{ind}.pkl", "wb") as f:
            pickle.dump(posterior, f)

        with open(save_dir / f"post_samples_run_{ind}.pkl", "wb") as f:
            pickle.dump(post_samples, f)


    with open(save_dir / f"train_times.pkl", "wb") as f:
        pickle.dump(train_times, f)
    with open(save_dir / f"posterior_times.pkl", "wb") as f:
        pickle.dump(posterior_times, f)

       


class GNKEmbedding(torch.nn.Module):
        def __init__(self, n_obs, d_x, hidden_dim=20, output_dim=20):
            super().__init__()
            self.n_obs = n_obs
            self.d_x = d_x
            self.phi = torch.nn.Sequential(
                torch.nn.Linear(d_x, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU()
            )
            # Aggregate and produce final summary
            self.rho = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, output_dim)
            )
        def forward(self, x):
            batch_size = x.shape[0]
            x_flat = x.reshape(-1, self.d_x)
            embeddings = self.phi(x_flat)
            embeddings = embeddings.reshape(batch_size, self.n_obs, -1)
            aggregated = embeddings.mean(dim=1) 

            # Final summary: (batch, hidden_dim) -> (batch, output_dim)
            stat = self.rho(aggregated)
            return None, stat

if __name__ == "__main__":
    # Get config_name from command line argument
    config_name = sys.argv[1]
    lambda_value = sys.argv[2] if len(sys.argv) > 2 else 1.0
    lambda_value = float(lambda_value)
    main(config_name, lambda_value)

