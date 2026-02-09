"""Run gnk example."""

import sys
import os
import types

import jax
import jax.numpy as jnp

import argparse
import pickle as pkl

import arviz as az  # type: ignore
from jax import random

from functools import partial
import scipy.io

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rsnl.inference import run_rsnl, run_snl
from rsnl.model import get_standard_model, get_robust_model
# from rsnl.visualisations import plot_and_save_all


from jax._src.prng import PRNGKeyArray  # for typing
from typing import Optional
import numpyro.distributions as dist

from pathlib import Path
from omegaconf import OmegaConf
import time

from experiment_helper import sample_gandk_fully_reparameterized_jax, summary_gnk_data, get_prior_gnk, get_prior_sir, sir_simulator_combined_jax, summary_fun_sir





def run_gnk(config_name: str, seed: int, save_dir, data_dir, ROBUST_FLAG, MORE_BUDGET):

    config_path = Path(__file__).resolve().parent.parent.parent.parent / "rca_sbi" / "config"
    cfg = OmegaConf.load(config_path / f"{config_name}.yaml")

    if MORE_BUDGET:
        print("Running with more budget")
        num_sims_per_round = 1000
        num_rounds = 5
        num_final_posterior_samples = 5000
    else:
        print("Running for fair comparison")
        num_sims_per_round = 500
        num_rounds = 2
        num_final_posterior_samples = cfg.num_posterior_samples

    model = get_robust_model if ROBUST_FLAG else get_standard_model
    prior = get_prior_gnk() if config_name == "gnk" else get_prior_sir()
    rng_key = random.PRNGKey(seed)
    rng_key, sub_key1, sub_key2 = random.split(rng_key, 3)
    sim_fn = sample_gandk_fully_reparameterized_jax if config_name == "gnk" else sir_simulator_combined_jax
    sum_fn = summary_gnk_data if config_name == "gnk" else summary_fun_sir
    # true_params = jnp.array(cfg.theta_true)

    # load x_obs
    path_x_obs = data_dir / f"x_obs_mis_{seed}.pkl" if ROBUST_FLAG else data_dir / f"x_obs_{seed}.pkl"
    t_x_obs = pkl.load(open(path_x_obs, 'rb'))
    x_obs = jnp.array(t_x_obs.numpy()).squeeze()
    x_obs = summary_gnk_data(x_obs) if config_name == "gnk" else summary_fun_sir(x_obs)

    if ROBUST_FLAG:
        print("Running RSNL")
        mcmc = run_rsnl(model, prior, sim_fn, sum_fn, rng_key, x_obs,
                       jax_parallelise=False, true_params=None,
                       theta_dims = 4,
                       num_sims_per_round = num_sims_per_round,
                       num_rounds = num_rounds,
                       num_final_posterior_samples = num_final_posterior_samples,
                       num_chains = cfg.num_chains,
                       thinning = cfg.thin,
                       num_warmup = cfg.warmup_steps)

    else:
        print("Running SNL")
        mcmc = run_snl(model, prior, sim_fn, sum_fn, rng_key, x_obs,
                    jax_parallelise=False, true_params=None,
                    theta_dims = 4,
                    num_sims_per_round = num_sims_per_round,
                    num_rounds = num_rounds,
                    num_final_posterior_samples = num_final_posterior_samples,
                    num_chains = cfg.num_chains,
                    thinning = cfg.thin,
                    num_warmup = cfg.warmup_steps)

    mcmc.print_summary()
    inference_data = az.from_numpyro(mcmc)

    posterior_samples = inference_data.posterior.theta.to_numpy().squeeze() # (num_posterior_samples, theta_dims) np.array

    if ROBUST_FLAG:
        posterior_samples_gamma = inference_data.posterior.adj_params.to_numpy().squeeze() # (num_posterior_samples, 1) np.array
        with open(save_dir/ f"post_samples_gamma_run_{seed}.pkl", 'wb') as f:
            pkl.dump(posterior_samples_gamma, f)

    with open(save_dir/ f"post_samples_run_{seed}.pkl", 'wb') as f:
        pkl.dump(posterior_samples, f)


def main(config_name: str, ROBUST_FLAG, MORE_BUDGET):
    project_root = Path(__file__).resolve().parent.parent.parent.parent

    robust_str = "" if ROBUST_FLAG else "_well_specified"
    more_budget_str = "_more_budget" if MORE_BUDGET else ""

    save_name = "robust_snle" + robust_str + more_budget_str
    save_dir = project_root / "baselines" / "results" / config_name / save_name
    print(f"Saving results to {save_dir}")
    data_dir = project_root / "data" / config_name
    save_dir.mkdir(parents=True, exist_ok=True)

    posterior_times = []
    num_repeat = 5 if MORE_BUDGET else 20

    for i in range(num_repeat):
        print(f"\n{'='*60}")
        print(f"Run {i+1}/{num_repeat}")
        print(f"{'='*60}")
        t0 = time.time()
        run_gnk(config_name, seed = i, save_dir=save_dir, data_dir=data_dir, ROBUST_FLAG=ROBUST_FLAG, MORE_BUDGET=MORE_BUDGET)
        t1 = time.time()
        posterior_time = t1 - t0
        posterior_times.append(posterior_time)
    with open(save_dir / "posterior_times.pkl", "wb") as f:
        pkl.dump(posterior_times, f)


if __name__ == '__main__':
    # numpyro.set_host_device_count(4)
    config_name = sys.argv[1]
    robust_flag = sys.argv[2].lower() in ('true', '1', 'yes')
    more_budget_flag = sys.argv[3].lower() in ('true', '1', 'yes')
    print(f"Running {config_name} with robust flag {robust_flag} and more budget flag {more_budget_flag}")
    main(config_name, robust_flag, more_budget_flag)