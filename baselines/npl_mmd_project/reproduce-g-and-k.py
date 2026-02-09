import sys
from pathlib import Path
sys.path.append("./src")

# Add project root to path for config loading
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# from utils import sample_gandk_outl, k, MMD_approx
# from plot_functions import plot_gnk, SeabornFig2Grid
import NPL
# import NPL_prior
import models
import numpy as np
# import pandas as pd
# from scipy import stats
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import time
import pickle
import torch
# Before running: 
# 1) Set paths 
# 2) Indicate whether you want a new dataset or to load existing one 
# 3) Experiments are run for multiple runs - index which run you want plots for

# Get config_name from command line or default to 'gnk'
config_name = sys.argv[1]
# experiment_name is the same as config_name
experiment_name = config_name

# Set paths based on experiment_name
data_path = str(project_root / "data" / experiment_name)
results_path = f"../results/{experiment_name}/npl_mmd/"

# Create results directory if it doesn't exist
Path(results_path).mkdir(parents=True, exist_ok=True)

# Set to True to generate and save new datasets or False to load saved datasets
sample_data_bool = False

# Set model 
model_name = 'gandk' 
n = 100 # number of observations
d = 1 # dimension of data
# theta_star = np.array([3,1,1,-np.log(2)]) # true parameter value 
outl = 1 # number of different percentages of outliers to run for
l = -1  # kernel lengthscale
p = 4   # number of unknown parameters
R = 20 # number of independent runs
s = 1 # std of Gaussian data

# simulation_budget = 100000
B = 500 # number of posterior samples
m = 20 # number of samples within NPL
Nstep = 10 # number of iteration for the optimisation to find minimiser of MMD
eta = 1.0
# assert B * m * Nstep == simulation_budget, "Simulation budget must be equal to B * m * Nstep"
model = models.g_and_k_model(m, d)

#######################################################################
###### Replace datasets here with your own datasets if you want  
### datasets needs to be of dimensions R x outl x n where R is number of datasets, outl is number of outlier settings and n is number of observations

# Load data
datasets = np.zeros((R,outl,n))
for j in range(R):
    with open(data_path+'/x_obs_mis_{}.pkl'.format(j), 'rb') as f:
        X = pickle.load(f)
    # Convert torch tensor to numpy if needed
    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()
    # X should be shape (n, 1)
    if X.shape == (n, 1):
        datasets[j,0,:] = X[:, 0]
    elif X.shape == (n,):
        datasets[j,0,:] = X
    else:
        datasets[j,0,:] = X.reshape(n,)[:n]

print("Datasets loaded")
######################################################################  

# Obtain and save results 
if __name__=='__main__':
    times = []
    # summary_stats = np.zeros((R,outl, p, 4)) # collect mean, median, mode, st.dev for each bootstrap sample
    for j in range(R):
        print("-----Run ", j, flush=True)
        for n_cont in range(outl):
            # print("-----Running for", n_cont*5, "% of outliers-----")
            X =datasets[j,n_cont,:].reshape((n,1))
            npl = NPL.npl(X,B,m,p,l, Nstep = Nstep, eta = eta, model = model, model_name = model_name)
            t0 = time.time()
            npl.draw_samples()
            t1 = time.time()
            total = t1 - t0
            times.append(total)
            sample = npl.sample # [B, p]
            sample[:, 1] = np.log(sample[:, 1]) # log(b)
            sample[:, 3] = np.log(sample[:, 3]) # log(k)
            
            with open(results_path+'/post_samples_run_{}.pkl'.format(j), 'wb') as f:
                pickle.dump(sample, f)
            
    with open(results_path+'/posterior_times.pkl', 'wb') as f:
        pickle.dump(times, f)
