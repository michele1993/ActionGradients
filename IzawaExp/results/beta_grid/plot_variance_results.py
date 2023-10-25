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
        outcome_dir = os.path.join(acc_dir,label+'_outcome_variability.npy') # For the mixed model
        var = np.load(outcome_dir)
        per_s_var.append(var)
    tot_var.append(np.array(per_s_var))
 
tot_var = np.array(tot_var) / 0.0176 # convert to degrees
mean_var = np.mean(tot_var, axis=0)
std_var = np.std(tot_var, axis=0)

font_s =7
mpl.rc('font', size=font_s)
plt.rcParams["font.family"] = "helvetica"
mpl.rcParams['xtick.labelsize'] = font_s 
mpl.rcParams['ytick.labelsize'] = font_s 

conditions = ["EBL","Mixed","RBL"]

# Subplot
fig, ax2 = plt.subplots(nrows=1, ncols=3, figsize=(5.5,3),
 gridspec_kw={'wspace': 0.5, 'hspace': 0.3, 'left': 0.1, 'right': 0.95, 'bottom': 0.15,
                                               'top': 0.95})
EBL_mean_variability = mean_var[-1]
EBL_std_variability = std_var[-1]

## Based on Izawa's results, use beta=0.3 
Mixed_mean_variability = mean_var[2]
Mixed_std_variability = std_var[2]

RBL_mean_variability = mean_var[0]
RBL_std_variability = std_var[0]

outcome_variabilities = [EBL_mean_variability, Mixed_mean_variability, RBL_mean_variability]
outcome_variabilities_std = [EBL_std_variability, Mixed_std_variability, RBL_std_variability]

## PLot the bar chart across 3 conditions
ax2[0].bar(conditions,outcome_variabilities,align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k') #color='tab:gray',
ax2[0].errorbar(conditions,outcome_variabilities, yerr=outcome_variabilities_std, ls='none', color='black',  elinewidth=1, capsize=1.5) # ecolor='lightslategray',
ax2[0].set_ylim([0, 4])
ax2[0].set_yticks([0,1,2,3,4])
ax2[0].spines['right'].set_visible(False)
ax2[0].spines['top'].set_visible(False)
ax2[0].set_ylabel('Reach Variability [deg]')

# Plot 'continous' change in variability across beta values
ax2[1].errorbar(betas,mean_var, yerr=std_var,capsize=3, fmt="r--o", ecolor = "black")
ax2[1].set_ylim([1.5, 4])
ax2[1].set_yticks([2,3,4])
ax2[1].spines['right'].set_visible(False)
ax2[1].spines['top'].set_visible(False)
ax2[1].set_xlabel('Beta values')


## Plot generalisation results
# Load generalisation results
file_dir = os.path.join(file_dir,'..','generalisation','gener_results.npy')
generalisation_res = np.load(file_dir)
gen_mean = np.flip(generalisation_res[0,:])
gen_std = np.flip(generalisation_res[1,:])

ax2[2].errorbar(conditions, gen_mean, yerr=gen_std,capsize=3, fmt="o", ecolor = "black")
ax2[2].spines['right'].set_visible(False)
ax2[2].spines['top'].set_visible(False)
ax2[2].set_ylabel('Generalisation error')


#plt.show()
## SAVE: figure
#plt.savefig('/Users/px19783/Desktop/LinearMotorModel', format='png', dpi=1200)
