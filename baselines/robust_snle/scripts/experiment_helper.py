import sys
import os

import jax
import jax.numpy as jnp

from jax import random


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from rsnl.visualisations import plot_and_save_all

from jax._src.prng import PRNGKeyArray  # for typing
import numpyro.distributions as dist


def sample_gandk_fully_reparameterized_jax(rng_key: PRNGKeyArray, 
                                        A: jnp.ndarray,
                                        logB: jnp.ndarray,
                                        g: jnp.ndarray,
                                        logk: jnp.ndarray,
                                        n: int = 100):
    """
    Jax implementation of sample_gandk_fully_reparameterized function
    
    Args:
        rng_key: JAX PRNG key for random number generation
        A: Location parameter
        logB: Log of scale parameter
        g: Skewness parameter
        logk: Log of kurtosis parameter
        n: Number of samples to generate (default 100)
    
    Returns:
        jnp.ndarray: Sampled data point(s) from the g-and-k distribution
    """
    
    # Transform constrained parameters back to their original space
    B = jnp.exp(logB)
    k = jnp.exp(logk)

    # Use JAX random number generation (not jnp.randn which doesn't exist)
    z = random.normal(rng_key, shape=(n,))
    term_g = jnp.tanh(0.5 * g * z)
    term_k = (1 + z**2)**k
    x = A + B * (1 + 0.8 * term_g) * term_k * z

    return x

def get_prior_gnk():
    """Return prior """
    prior_mean = jnp.array([0.0, 0.7, 0.0, -1.5], dtype=jnp.float32)
    prior_cov = jnp.array([
        [5.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 4.0, 0.0],
        [0.0, 0.0, 0.0, 0.25]
    ], dtype=jnp.float32)
    prior = dist.MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    return prior

def summary_gnk_data(x): 
    """
    x is a JAX array of shape (n_obs,)
    summary_stats is a JAX array of shape (summary_dim = 4,)
    """
    s1 = jnp.mean(x, axis=-1, keepdims=True)
    s2 = jnp.std(x, axis=-1, keepdims=True)
    s3 = jnp.mean(x ** 3, axis=-1, keepdims=True)
    s4 = jnp.mean(x ** 4, axis=-1, keepdims=True)
    summary_stats = jnp.concatenate([s1, s2, s3, s4], axis=-1)
    return summary_stats


## TODO:

def get_prior_sir():
    """Return prior for SIR model parameters.
    
    Parameters are: [log_beta, log_gamma, logit_rho, log_I0]
    Based on config: prior_mean = [0.5, 0.2, 0.5, 20] (in natural space)
    prior_std = [0.5, 0.5, 1.0, 0.7] (in transformed space)
    
    Transformations:
    - beta = 0.5 -> log_beta = log(0.5)
    - gamma = 0.2 -> log_gamma = log(0.2)
    - rho = 0.5 -> logit_rho = logit(0.5) = 0.0
    - I0 = 20 -> log_I0 = log(20)
    """
    # Transform from natural space to transformed space
    beta_mean = 0.5
    gamma_mean = 0.2
    rho_mean = 0.5
    I0_mean = 20.0
    
    # logit(x) = log(x / (1 - x))
    logit_rho = jnp.log(rho_mean / (1.0 - rho_mean))
    
    prior_mean = jnp.array([
        jnp.log(beta_mean),         # log beta
        jnp.log(gamma_mean),        # log gamma
        logit_rho,                  # logit rho
        jnp.log(I0_mean),           # log I0
    ], dtype=jnp.float32)
    
    prior_cov = jnp.array([
        [0.25, 0.0, 0.0, 0.0],      # 0.5^2
        [0.0, 0.25, 0.0, 0.0],      # 0.5^2
        [0.0, 0.0, 1.0, 0.0],       # 1.0^2
        [0.0, 0.0, 0.0, 0.49]       # 0.7^2
    ], dtype=jnp.float32)
    prior = dist.MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    return prior

def sir_simulator_jax(rng_key: PRNGKeyArray,
                  log_beta: jnp.ndarray,
                  log_gamma: jnp.ndarray,
                  logit_rho: jnp.ndarray,
                  log_I0: jnp.ndarray,
                  T: int = 150,
                  N: int = 1000):
    """
    Pure JAX implementation of SIR simulator.
    
    Stochastic discrete-time SIR with Binomial transitions and Poisson observations.
    
    Args:
        rng_key: JAX PRNG key for random number generation
        log_beta: Log transmission rate (scalar or 0-d array)
        log_gamma: Log recovery rate (scalar or 0-d array)
        logit_rho: Logit reporting rate (scalar or 0-d array)
        log_I0: Log initial infected count (scalar or 0-d array)
        T: Number of time steps (default 150)
        N: Population size (default 1000)
    
    Returns:
        jnp.ndarray: Observed new cases per day, shape (T,)
    """
    # Convert to scalars if needed
    log_beta = float(log_beta) if hasattr(log_beta, 'item') else float(log_beta)
    log_gamma = float(log_gamma) if hasattr(log_gamma, 'item') else float(log_gamma)
    logit_rho = float(logit_rho) if hasattr(logit_rho, 'item') else float(logit_rho)
    log_I0 = float(log_I0) if hasattr(log_I0, 'item') else float(log_I0)
    
    # Transform parameters
    beta = jnp.exp(log_beta)
    gamma = jnp.exp(log_gamma)
    rho = jax.nn.sigmoid(logit_rho)
    I0 = jnp.clip(jnp.round(jnp.exp(log_I0)), a_min=1.0)
    
    # Initialize state
    S0 = float(N) - I0
    I0_val = I0
    R0 = 0.0
    
    dt = 1.0
    
    # Split RNG key for each time step
    rng_keys = random.split(rng_key, T)
    
    def step(carry, rng_t):
        S_prev, I_prev, R_prev = carry
        
        # Clamp to ensure non-negative
        St = jnp.clip(S_prev, a_min=0.0)
        It = jnp.clip(I_prev, a_min=0.0)
        
        # Infection probability over dt (mass-action)
        # p_inf = 1 - exp(-beta * I/N * dt)
        p_inf = 1.0 - jnp.exp(-beta * (It / float(N)) * dt)
        p_inf = jnp.clip(p_inf, 0.0, 1.0)
        
        # Recovery probability over dt
        # p_rec = 1 - exp(-gamma * dt)
        p_rec = 1.0 - jnp.exp(-gamma * dt)
        p_rec = jnp.clip(p_rec, 0.0, 1.0)
        
        # Sample from binomial distributions
        # JAX's binomial uses n (total_count) and p (probs)
        rng_inf, rng_rec, rng_obs = random.split(rng_t, 3)
        
        # Binomial: new_inf ~ Binomial(St, p_inf)
        new_inf = random.binomial(rng_inf, n=St.astype(jnp.int32), p=p_inf).astype(jnp.float32)
        
        # Binomial: new_rec ~ Binomial(It, p_rec)
        new_rec = random.binomial(rng_rec, n=It.astype(jnp.int32), p=p_rec).astype(jnp.float32)
        
        # Update states
        S_new = St - new_inf
        I_new = It + new_inf - new_rec
        R_new = R_prev + new_rec
        
        # Observations: reported incident infections
        mu = jnp.clip(rho * new_inf, a_min=0.0)
        
        # Poisson observation model
        y_t = random.poisson(rng_obs, mu).astype(jnp.float32)
        
        return (S_new, I_new, R_new), y_t
    
    # Run simulation
    initial_carry = (S0, I0_val, R0)
    _, y = jax.lax.scan(step, initial_carry, rng_keys)
    
    return y

def sir_summary_jax(x: jnp.ndarray, N: float = 1000.0):
    """
    Pure JAX implementation of SIR summary statistics.
    
    Args:
        x: JAX array of shape (T,) or (n, T) - observed new cases per day
        N: Population size (default 1000)
    
    Returns:
        jnp.ndarray: Summary statistics [attack_rate, t_peak_scaled, peak_scaled]
                    Shape (3,) if x is 1D, or (n, 3) if x is 2D
    """
    # Handle both 1D and 2D inputs
    if x.ndim == 1:
        x = x[None, :]  # Add batch dimension
        squeeze_output = True
    else:
        squeeze_output = False
    
    n, T = x.shape
    
    # Attack rate: sum of cases / N
    attack = jnp.sum(x, axis=1) / float(N)
    
    # Peak: max cases / N
    peak = jnp.max(x, axis=1) / float(N)
    
    # Time of peak (scaled by T-1)
    t_peak = jnp.argmax(x, axis=1).astype(jnp.float32) / float(max(T - 1, 1))
    
    summary = jnp.stack([attack, t_peak, peak], axis=1)
    
    if squeeze_output:
        summary = summary[0]  # Remove batch dimension
    
    return summary

def sir_simulator_combined_jax(rng_key: PRNGKeyArray,
                               log_beta: jnp.ndarray,
                               log_gamma: jnp.ndarray,
                               logit_rho: jnp.ndarray,
                               log_I0: jnp.ndarray,
                               T: int = 150,
                               N: int = 1000):
    """
    Combined SIR simulator that returns summary statistics directly.
    
    This function combines simulation and summary computation in one step,
    which is useful for inference workflows that need summary statistics.
    
    Args:
        rng_key: JAX PRNG key (not used directly, but kept for interface consistency)
        log_beta: Log transmission rate (scalar or 0-d array)
        log_gamma: Log recovery rate (scalar or 0-d array)
        logit_rho: Logit reporting rate (scalar or 0-d array)
        log_I0: Log initial infected count (scalar or 0-d array)
        T: Number of time steps (default 150)
        N: Population size (default 1000)
    
    Returns:
        jnp.ndarray: Summary statistics [attack_rate, t_peak_scaled, peak_scaled], shape (3,)
    """
    # Simulate SIR model
    y = sir_simulator_jax(rng_key, log_beta, log_gamma, logit_rho, log_I0, T=T, N=N)
    
    # Compute summary statistics
    summary = sir_summary_jax(y, N=N)
    
    return summary


def summary_fun_sir(x: jnp.ndarray):
    """
    x is a jax array of [n_obs, 3]
    """
    mean_vec = x.mean(axis=0)
    std_vec = x.std(axis=0)
    return jnp.concatenate([mean_vec, std_vec], axis=-1)