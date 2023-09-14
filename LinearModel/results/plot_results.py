import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import set_matplotlib_formats

## Load data
file_dir = os.path.dirname(os.path.abspath(__file__))

label_1 = 'EBL'
label_2 = 'Mixed'
label_3 = 'RBL'

labels = [label_1, label_2, label_3]
results = []

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

for l in labels:
    results.append(np.load(os.path.join(file_dir,l+'_outcome_variability.npy')))

results = np.array(results).squeeze() / 0.0176 # convert to degrees
font_s =7
mpl.rc('font', size=font_s)
plt.rcParams["font.family"] = "helvetica"
mpl.rcParams['xtick.labelsize'] = font_s 
mpl.rcParams['ytick.labelsize'] = font_s 

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(7.5,3),
 gridspec_kw={'wspace': 0.32, 'hspace': 0.3, 'left': 0.1, 'right': 0.95, 'bottom': 0.15,
                                               'top': 0.95})

i=0
t = np.arange(1,results.shape[-1]+1)
for d in results:
    axs[i].plot(t,d,c='k',linewidth=0.5)
    # plot required adaptation
    axs[i].plot(t,optimal_adaptation,c='yellow',linewidth=1)
    axs[i].set_ylim([-10, 20])
    #axs[i].set_title(labels[i],fontsize=font_s)
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)
    axs[i].set_xlabel('Trials')
    if i == 0:
        axs[i].set_ylabel('Reach Angle [deg]')
    i+=1

## SAVE: 1st figure
#plt.savefig('/Users/px19783/Desktop/ActionGrad_1st_experiment_accuracy', format='png', dpi=1200)

# Subplot
fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(2.5,3.5),
 gridspec_kw={'wspace': 0.32, 'hspace': 0.3, 'left': 0.2, 'right': 0.95, 'bottom': 0.15,
                                               'top': 0.95})
fixed_trials =100
EBL_outcome_variability = np.std(results[0,-fixed_trials:])
Mixed_outcome_variability = np.std(results[1,-fixed_trials:])
RBL_outcome_variability = np.std(results[2,-fixed_trials:])
outcome_variabilities = [EBL_outcome_variability, Mixed_outcome_variability, RBL_outcome_variability]
conditions = ['ERR','EPE', 'RWD']
for d in results:
    ax2.bar(conditions,outcome_variabilities,align='center', alpha=0.5,ecolor='black', capsize=5, color='tab:gray',edgecolor='k')
    ax2.set_ylim([0, 4])
    ax2.set_yticks([0,1,2,3,4])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_ylabel('Reach Variability [deg]')

plt.show()
## SAVE: 2nd figure
#plt.savefig('/Users/px19783/Desktop/ActionGrad_1st_experiment_variability', format='png', dpi=1200)
