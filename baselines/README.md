This directory contains code to run baselines and visualise results.

Baseline methods implemented are:
- Daolang Huang, Ayush Bharti, Amauri H. Souza, Luigi Acerbi, and Samuel Kaski. 2023. Learning robust statistics for simulation-based inference under model misspecification. In Proceedings of the 37th International Conference on Neural Information Processing Systems
- Dellaporta, Charita, et al. "Robust Bayesian inference for simulator-based models via the MMD posterior bootstrap." International Conference on Artificial Intelligence and Statistics. PMLR, 2022.
- Gao, R., Deistler, M., & Macke, J. H. (2023). Generalized bayesian inference for scientific simulators via amortized cost estimation. Advances in Neural Information Processing Systems, 36, 80191-80219.


## Running baselines

### Learning robust statistics for simulation-based inference under model misspecification. 

1. Change directory and create/activate conda environment

Change directory with `cd Robust-SBI` and follow an instruction in `Robust-SBI/README.md` to create an environnment and activate with
```conda activate Robust-SBI```.

2. Run experiments

Experiment with g-and-k simulator is done by
```python exp_gnk.py```
We follow the configuration (prior hyperparameres, number of runs etc) from `rca_sbi/config/gnk.yaml`. We deactivate the environment with `conda deactivate`.

### Robust Bayesian inference for simulator-based models via the MMD posterior bootstrap

1. Change directory and create / activate an environment 

Change directory with `cd npl_mmd_project` and create an environment using `npl_mmd_project/environment.yaml`.
I created an environemnt with uv. We can activate with
```source .venv/bin/activate```

2. Run experiments
Experiment with g-and-k simulator is done by
```python reproduce-g-and-k.py```
Note that this method does not require prior specification. All the hyperparameters are defined inside `reproduce-g-and-k.py`. 

The hyperparameters needs to be consistent with other method is `R` (number of runs), `n` (number of i.i.d observation) `p` (dimension of parameter) `d` (dimension of data), and `B` (number of posterior samples). 
Sampling with cpu takes enormousm time so I am considering to change code such that it only runs once on cpu to measure time, and switch to gpu afterwards. We deactivate the environment with `deactivate`.

### Generalized bayesian inference for scientific simulators via amortized cost estimation.
1. Change directory `cd neuralgbi` and create / activate an enviromenmt following readme file of the project.
2. Run experiment

Experiment with g-and-k simulator is done by
```python exp_gnk.py```


## data
The data directory contains data used for evaluation.

### gnk
20 observation with outliers (`x_obs_mis_{idx}.pkl`), one clean observation (`x_obs_ref.pt`), and reference posterior samples `sample_nle_ref.pt`.

## evaluations
The evaluatoin directory contains files to evaluate the result.
For example, to evaluate 0th run of the two method, follow:
``` python evaluate_gnk.py 0 npl_mmd ``` 
``` python evaluate_gnk.py 0 robust_summary```

It produces mmd, c2st, and some visualisation of the result. Since mmd and c2st is calculated for each run, I should write a code to aggregate them.

## results
The results directory contains posterior samples, c2st, and mmd produced from the evaluations.
Additional files such as training data is also saved here.

## figures
The figures directory contains the visualisation of resulting marginal posterior, reference marginal posterior, and histgram of observed data.g
