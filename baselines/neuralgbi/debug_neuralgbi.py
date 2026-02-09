from pathlib import Path
import pickle
import argparse
project_root = Path(__file__).resolve().parent.parent.parent
from experiment_helper import compute_mmd_lengthscale
from gbi.distances import mmd_dist
from functools import partial
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf


import matplotlib

matplotlib.rcParams.update({
    'font.family' : 'serif',
    'font.size' : 14.0,
    'lines.linewidth' : 2,
    'lines.antialiased' : True,
    'axes.facecolor': 'fdfdfd',
    'axes.edgecolor': '777777',
    'axes.linewidth' : 1,
    'axes.titlesize' : 'medium',
    'axes.labelsize' : 'medium',
    'axes.axisbelow' : True,
    'xtick.major.size'     : 0,      # major tick size in points
    'xtick.minor.size'     : 0,      # minor tick size in points
    'xtick.major.pad'      : 6,      # distance to major tick label in points
    'xtick.minor.pad'      : 6,      # distance to the minor tick label in points
    'xtick.color'          : '333333', # color of the tick labels
    'xtick.labelsize'      : 'medium', # fontsize of the tick labels
    'xtick.direction'      : 'in',     # direction: in or out
    'ytick.major.size'     : 0,      # major tick size in points
    'ytick.minor.size'     : 0,      # minor tick size in points
    'ytick.major.pad'      : 6,      # distance to major tick label in points
    'ytick.minor.pad'      : 6,      # distance to the minor tick label in points
    'ytick.color'          : '333333', # color of the tick labels
    'ytick.labelsize'      : 'medium', # fontsize of the tick labels
    'ytick.direction'      : 'in',     # direction: in or out
    'axes.grid' : False,
    'grid.alpha' : 0.3,
    'grid.linewidth' : 1,
    'legend.fancybox' : True,
    'legend.fontsize' : 'Small',
    'figure.figsize' : (2.5, 2.5),
    'figure.facecolor' : '1.0',
    'figure.edgecolor' : '0.5',
    'hatch.linewidth' : 0.1,
    'text.usetex' : True
    })


parser = argparse.ArgumentParser(description="Debug NeuralGBI distance prediction")
parser.add_argument("--config_name", type=str, required=True, help="Configuration name (e.g., sir_undercounting, gnk, turin)")
parser.add_argument("--index", type=int, required=True, help="Index for the run")
args = parser.parse_args()

contaminated_obs_flag = True
more_budget_flag = False
config_name = args.config_name
index = args.index

config_path = Path(__file__).resolve().parent.parent.parent / "rca_sbi" / "config"
inference_path = project_root / "baselines" / "results" / config_name / "neuralgbi" / fr"inference_run_{index}.pkl" if not more_budget_flag else project_root / "baselines" / "results" / config_name / "neuralgbi_more_budget" / fr"inference_run_{index}.pkl"
save_dir = inference_path.parent
cfg = OmegaConf.load(config_path / f"{config_name}.yaml")
dtype = torch.float32


inference = pickle.load(open(inference_path, "rb"))



if config_name in ["sir", "sir_undercounting", "sir_student_t_1", "sir_student_t_2"]:
    theta_true_orig = torch.tensor(cfg.theta_true, dtype = dtype)
    theta_true = torch.tensor([
    torch.log(torch.tensor(theta_true_orig[0])),         # log beta
    torch.log(torch.tensor(theta_true_orig[1])),        # log gamma
    torch.logit(torch.tensor(theta_true_orig[2])),        # logit rho
    torch.log(torch.tensor(theta_true_orig[3])),           # log I0
    ])
    training_data_dir = project_root / "data" / "sir" # training data is same all the time.
    data_dir = project_root / "data" / config_name

elif config_name in ["gnk", "turin"]:
    theta_true = torch.tensor(cfg.theta_true, dtype = dtype)
    data_dir = project_root / "data" / config_name
    training_data_dir = data_dir


if more_budget_flag:
    x_sim = torch.load(training_data_dir / f"x_sim_{index}_ace_more_budget.pt")
    theta = torch.load(training_data_dir / f"theta_{index}_ace_more_budget.pt")
else:
    x_sim = torch.load(training_data_dir / f"x_sim_{index}_ace.pt")
    theta = torch.load(training_data_dir / f"theta_{index}_ace.pt")

if contaminated_obs_flag:
    x_obs = pickle.load(open(data_dir / f"x_obs_mis_{index}.pkl", "rb"))
else:
    x_obs = x_sim[0]
    theta_true = theta[0]

lengthscale = compute_mmd_lengthscale(x_obs)
distance_func = partial(mmd_dist, lengthscale=lengthscale)
dist_array = np.zeros((x_sim.shape[0],2))

for i in range(x_sim.shape[0]):
    true_dist = distance_func(x_sim[i].unsqueeze(0).unsqueeze(0), x_obs.unsqueeze(0).unsqueeze(0))
    predicted_dist = inference.distance_net(theta[i].unsqueeze(0), x_obs.unsqueeze(0))
    dist_array[i,0] = true_dist.item()
    dist_array[i,1] = predicted_dist.item()

true_dist_theta_true = distance_func(x_obs.unsqueeze(0).unsqueeze(0), x_obs.unsqueeze(0).unsqueeze(0))
predicted_dist_theta_true = inference.distance_net(theta_true.unsqueeze(0), x_obs.unsqueeze(0))


fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(dist_array[:,0], dist_array[:,1])
ax.scatter(true_dist_theta_true.item(), predicted_dist_theta_true.item(), color='red', marker='x', s=100, label='True theta')
ax.set_xlabel("True distance")
ax.set_ylabel("Predicted distance")
ax.set_title("")
ax.grid(True)
fig.savefig(save_dir / f"distance_prediction_{index}.pdf", bbox_inches='tight')
exit()







