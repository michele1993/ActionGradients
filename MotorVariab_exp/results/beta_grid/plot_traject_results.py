import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import set_matplotlib_formats

## Load data
seeds = [8721, 5467, 1092, 9372,2801]
betas = np.arange(0,11,1) /10.0
file_dir = os.path.dirname(os.path.abspath(__file__))

tot_var = []
for s in seeds:
    acc_dir = os.path.join(file_dir,str(s))
    per_s_var = []
    for b in betas:
        label = "Mixed_"+str(b)
        outcome_dir = os.path.join(acc_dir,label+'_trajectories.npy') # For the mixed model
        var = np.load(outcome_dir)
        per_s_var.append(var)
    tot_var.append(np.array(per_s_var))
 

tot_var = np.array(tot_var) / 0.0176 # convert to degrees
# Select trajectories for beta=0,0.3,1 (i.e., base on variance analysis resutls, best capturing Izawa's)
results = tot_var.mean(axis=0)[[0,2,10],:].squeeze()

# ===== Compute optimal amount of adaptation based on perturbation
n_trials_x_perturbation = 40
perturbs = np.arange(0,9)
optimal_adaptation = []
for p in perturbs:

 optimal_adaptation.append([p] * n_trials_x_perturbation)

## Add opt adapt for final trials with corresponding 8 degree perturbation
fixed_trials = 140 # Note 140 because last pertub had 40 extra

optimal_adaptation.append([perturbs[-1]] * fixed_trials)
optimal_adaptation = np.array(sum(optimal_adaptation,[])) 


font_s =7
mpl.rc('font', size=font_s)
plt.rcParams["font.family"] = "helvetica"
mpl.rcParams['xtick.labelsize'] = font_s 
mpl.rcParams['ytick.labelsize'] = font_s 

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(7.5,3),
 gridspec_kw={'wspace': 0.32, 'hspace': 0.3, 'left': 0.1, 'right': 0.95, 'bottom': 0.15,
                                               'top': 0.9})

conditions = ["EBL","Mixed","RBL"]
# Reverse order to plot in the same order as Izawa
results = np.flip(results,axis=0)
i=0
t = np.arange(1,results.shape[-1]+1)
for d in results:
    axs[i].plot(t,d,c='k',linewidth=0.5)
    # plot required adaptation
    axs[i].plot(t,optimal_adaptation,c='yellow',linewidth=1)
    axs[i].set_ylim([-10, 20])
    axs[i].set_title(conditions[i],fontsize=font_s)
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)
    axs[i].set_xlabel('Trials')
    if i == 0:
        axs[i].set_ylabel('Reach Angle [deg]')
    i+=1

## SAVE: 1st figure
#plt.savefig('/Users/px19783/Desktop/ActionGrad_1st_experiment_accuracy', format='png', dpi=1200)
plt.show()
