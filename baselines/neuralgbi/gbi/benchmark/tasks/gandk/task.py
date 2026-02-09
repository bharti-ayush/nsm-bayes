from typing import Optional
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal
import sys
from pathlib import Path

# Add project root to path to access rca_sbi simulators
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from rca_sbi.simulators import sample_gandk_fully_reparameterized


class GAndKTask:
    def __init__(
        self,
        prior_mean: Optional[Tensor] = None,
        prior_cov: Optional[Tensor] = None,
        seed: int = 0,
    ):
        """G-and-k task for neuralgbi.
        
        Args:
            prior_mean: Mean of the prior (4D: [A, log(B), g, log(k)])
            prior_cov: Covariance matrix of the prior (4x4)
            seed: Random seed
        """
        torch.manual_seed(seed)
        
        # Default prior
        if prior_mean is None:
            prior_mean = torch.tensor([0.0, 0.7, 0.0, -1.5], dtype=torch.float32)
        if prior_cov is None:
            prior_cov = torch.tensor([
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.25]
            ], dtype=torch.float32)
        
        self.prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
        self.d_theta = 4
        self.d_x = 1
        
    def simulate(self, theta: Tensor, n_obs: int = 100) -> Tensor:
        """Simulate g-and-k data for given parameters.
        
        Args:
            theta: (batch_size, 4) tensor of parameters [A, log(B), g, log(k)]
            n_obs: Number of iid observations per theta (default 100)
            
        Returns:
            x: (batch_size, n_obs, d_x) tensor of simulated data - n_obs x per theta
        """
        batch_size = theta.shape[0]
        x = torch.zeros(batch_size, n_obs, self.d_x)
        for i in range(batch_size):
            for j in range(n_obs):
                x[i, j, :] = sample_gandk_fully_reparameterized(theta[i, :], n=1)
        return x

