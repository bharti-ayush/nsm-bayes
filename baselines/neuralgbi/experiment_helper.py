"""
Experiment script for g-and-k simulator using NeuralGBI (GBI method).

This script follows the neuralgbi pattern but adapted for the baseline framework.
It trains GBI and runs inference for multiple runs, saving results in the same
format as other baselines.
"""

import torch
import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from gbi.GBI import GBInference
import gbi.utils.utils as gbi_utils

from sbi.inference import MCMCPosterior
from sbi.utils import mcmc_transform


from torch.distributions import MultivariateNormal


def compute_mmd_lengthscale(x: torch.Tensor) -> float:
    if x.dim() == 3:
        x = x.reshape(-1, x.size(-1))
    elif x.dim() == 2:
        x = x
    else:
        raise ValueError(f"Expected x with dim 2 or 3, got shape {x.shape}")

    # Need at least two points
    if x.size(0) < 2:
        return torch.tensor(1.0, dtype=x.dtype, device=x.device)

    # Pairwise Euclidean distances, then square
    dists = torch.pdist(x)          # [N*(N-1)/2]
    sq_dists = dists ** 2

    # Median heuristic: l = sqrt( median(||x_i - x_j||^2) / 2 )
    median_sq = sq_dists.median()
    lengthscale = torch.sqrt(median_sq / 2.0)

    return lengthscale.item() # return float


def train_GBI(theta, x, x_obs, task, distance_func, gbi_config):
    """Train GBI.
    
    Args:
        theta: Parameter samples (batch_size, d_theta)
        x: Simulated data (batch_size, n_obs, d_x) - n_obs x per theta
        x_obs: Observed data (n_obs, d_x) - n_obs observations
        task: GAndKTask instance
        distance_func: Distance function to use
        config: Configuration dict with GBI hyperparameters
        data_dir: Path to data directory
        
    Returns:
        inference: Trained GBInference object
    """
    # For mmd_dist, keep x in shape (batch_size, n_obs, d_x) for x_target
    # Following gaussian_mixture pattern: x_target keeps the (batch, n_obs, dim) shape
    batch_size, n_obs, d_x = x.shape
    
    # Augment subset of training data with noise.
    if gbi_config.get('n_augmented_x', 0) > 0:
        # Select random batches and add noise
        aug_indices = torch.randint(batch_size, size=(gbi_config['n_augmented_x'],))
        x_aug = x[aug_indices].clone()  # (n_augmented_x, n_obs, d_x)
        # Add noise to each observation
        x_aug = x_aug + torch.randn_like(x_aug) * x.std(dim=(0, 1), keepdim=True) * gbi_config.get('noise_level', 2.0)
        x_target = gbi_utils.concatenate_xs(x, x_aug)
    else:
        x_target = x

    # Load and append observed data if train_with_obs is True
    ## Here we load the observed data for the current run (ind)
    ## This is because we want every run to be independent, just like other baselinees
    ## If we load all the observed data (idx = 0,...,20), then this is not the case.
    # x_obs should be (n_obs, d_x), reshape to (1, n_obs, d_x) to match x_target shape
    if x_obs.dim() == 2:
        x_obs = x_obs.unsqueeze(0)  # (n_obs, d_x) -> (1, n_obs, d_x)
    x_target = gbi_utils.concatenate_xs(x_target, x_obs)

    # Initialize GBI
    inference = GBInference(
        prior=task.prior,
        distance_func=distance_func,
        do_precompute_distances=gbi_config.get('do_precompute_distances', False),
    )
    # For mmd_dist, x needs to be (batch_size, n_obs, d_x) to match the expected shape
    inference = inference.append_simulations(theta, x, x_target)
    
    # Initialize distance estimator
    # For g-and-k with mmd_dist and multiple observations, use PermutationInvariantEmbedding
    # Following gaussian_mixture pattern: trial_net_input_dim = d_x, trial_net_output_dim = 20
    net_kwargs = {"trial_net_input_dim": d_x, "trial_net_output_dim": 20}
    
    inference.initialize_distance_estimator(
        num_layers=gbi_config.get('num_layers', 3),
        num_hidden=gbi_config.get('num_hidden', 64),
        net_type=gbi_config.get('net_type', 'resnet'),
        positive_constraint_fn=gbi_config.get('positive_constraint_fn', 'softplus'),
        net_kwargs=net_kwargs,
    )
    
    # Train
    _ = inference.train(
        training_batch_size=gbi_config.get('training_batch_size', 500),
        max_n_epochs=gbi_config.get('max_epochs', 5000),
        validation_fraction=gbi_config.get('validation_fraction', 0.1),
        n_train_per_theta=gbi_config.get('n_train_per_theta', 2),
        n_val_per_theta=gbi_config.get('n_val_per_theta', 5),
        stop_after_counter_reaches=gbi_config.get('n_epochs_convergence', 100),
        print_every_n=gbi_config.get('print_every_n', 20),
        plot_losses=False,
    )
    
    return inference


def infer_GBI(inference, x_obs, prior, beta, num_posterior_samples, config):
    """Run inference (posterior sampling) for g-and-k task.
    
    Args:
        inference: Trained GBInference object
        x_obs: Observed data (n_obs, d_x) - n_obs observations
        prior: Prior distribution
        beta: Distance weight parameter
        num_posterior_samples: Number of posterior samples to generate
        config: config for rca_sbi
        
    Returns:
        post_samples: Posterior samples (num_posterior_samples, d_theta)
    """

    #Add dimension following run_inference.py (lines 142-145) for gaussian_mixture task
    x_obs = x_obs.unsqueeze(0) # [n_obs, d_x] -> [1, n_obs, d_x]

    potential_fn = inference.get_potential(x_o=x_obs, beta=beta)
    theta_transform = mcmc_transform(prior)
    
    posterior = MCMCPosterior(
        potential_fn,
        theta_transform=theta_transform,
        proposal=prior,
        method="slice_np_vectorized",
        thin=config.get('thin', 10),
        warmup_steps=config.get('warmup_steps', 50),
        num_chains=config.get('num_chains', 100),
        init_strategy="resample",
        frac_chains_to_finish=1.0,
    )
    
    post_samples = posterior.sample((num_posterior_samples,))
    print(f"Posterior samples shape: {post_samples.shape}")
    return post_samples


class GeneralTask:
    def __init__(
        self,
        prior_mean,
        prior_cov,
        d_theta,
        d_x,
    ):
        self.prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
        self.d_theta = d_theta
        self.d_x = d_x