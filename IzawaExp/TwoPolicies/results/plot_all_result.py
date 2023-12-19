import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import set_matplotlib_formats

## =============== Plot motor variability results ===============
## Load data
file_dir = os.path.dirname(os.path.abspath(__file__))

font_s =7
mpl.rc('font', size=font_s)
plt.rcParams["font.family"] = "helvetica"
mpl.rcParams['xtick.labelsize'] = font_s 
mpl.rcParams['ytick.labelsize'] = font_s 

# Subplot
fig, ax2 = plt.subplots(nrows=1, ncols=4, figsize=(6.5,2),
 gridspec_kw={'wspace': 0.5, 'hspace': 0.3, 'left': 0.1, 'right': 0.95, 'bottom': 0.25,
                                               'top': 0.95})
## ==== Plot training accuracy ====
seeds = [0, 9284, 5992, 7861, 1594]
sde = np.sqrt(len(seeds))
seed_NoBeta_acc = []
seed_NoBeta_a_std = []
seed_EBL_acc = []
seed_EBL_a_std = []
NoBeta_pretain = 1 # i.e. 100 trials of pretrain for NoBeta to ensure same start as EBL accuracy
for s in seeds:
    # Load accuracy
    NoBeta_acc_file = os.path.join(file_dir,str(s),'NoBeta_accuracy.npy')
    seed_NoBeta_acc.append(np.load(NoBeta_acc_file)[NoBeta_pretain:])
    EBL_acc_file = os.path.join(file_dir,str(s),'EBL_accuracy.npy')
    seed_EBL_acc.append(np.load(EBL_acc_file))

    # Load action std (for final trial)
    NoBeta_std_file = os.path.join(file_dir,str(s),'NoBeta_action_std.npy')
    seed_NoBeta_a_std.append(np.load(NoBeta_std_file))
    EBL_std_file = os.path.join(file_dir,str(s),'EBL_action_std.npy')
    seed_EBL_a_std.append(np.load(EBL_std_file))

seed_NoBeta_acc = np.array(seed_NoBeta_acc)
seed_EBL_acc = np.array(seed_EBL_acc)
seed_NoBeta_a_std = np.array(seed_NoBeta_a_std) /sde
seed_EBL_a_std = np.array(seed_EBL_a_std) /sde

NoBeta_acc_mean = seed_NoBeta_acc.mean(axis=0)
NoBeta_acc_std = seed_NoBeta_acc.std(axis=0) /sde
NoBeta_std_mean = seed_NoBeta_a_std.mean(axis=0)
NoBeta_std_std = seed_NoBeta_a_std.std(axis=0) /sde

EBL_acc_mean = seed_EBL_acc.mean(axis=0)
EBL_acc_std = seed_EBL_acc.std(axis=0) /sde
EBL_std_mean = seed_EBL_a_std.mean(axis=0)
EBL_std_std = seed_EBL_a_std.std(axis=0) /sde

t = np.arange(1,len(NoBeta_acc_mean)+1,1)
colors = ['tab:olive','tab:blue']
ax2[0].plot(t, NoBeta_acc_mean, alpha=0.75, color=colors[0],label="No $\\beta$-weighting")
ax2[0].fill_between(t,NoBeta_acc_mean-NoBeta_acc_std, NoBeta_acc_mean+NoBeta_acc_std, alpha = 0.25, color=colors[0])
ax2[0].plot(t, EBL_acc_mean, alpha=0.75, color=colors[1],label="$\\beta$-weighting")
ax2[0].fill_between(t,EBL_acc_mean-EBL_acc_std, EBL_acc_mean+EBL_acc_std, alpha = 0.25, color=colors[1])
ax2[0].spines['right'].set_visible(False)
ax2[0].spines['top'].set_visible(False)
ax2[0].set_ylabel('Training error')
ax2[0].set_xlabel('Trials (x100)')
ax2[0].xaxis.set_ticks_position('none') 
ax2[0].yaxis.set_ticks_position('none') 
ax2[0].legend(loc='upper left', bbox_to_anchor=(0.05, 1.0), frameon=False,fontsize=font_s)

## Plot variance for NoBeta and EBL
gen_mean = [NoBeta_std_mean, EBL_std_mean]
gen_std = [NoBeta_std_std, EBL_std_std]
conditions = ["No $\\beta$-weighting", "$\\beta$-weighting"]
for i, c in enumerate(colors):
    ax2[1].plot(conditions[i], gen_mean[i], 'o', markersize=5, color=c, alpha=0.5)
    ax2[1].errorbar(i, gen_mean[i], yerr=gen_std[i],capsize=3, ecolor = "black", elinewidth=0.75, color=c, fmt="None")
ax2[1].spines['right'].set_visible(False)
ax2[1].spines['top'].set_visible(False)
ax2[1].set_ylabel('Motor variability')
ax2[1].xaxis.set_ticks_position('none') 
ax2[1].yaxis.set_ticks_position('none') 


## ==== Plot two policies with changin betas ====
slow_cng_dir = os.path.join(file_dir,'Change_in_feedback_accuracy.npy')
sudden_cng_dir = os.path.join(file_dir,'Sudden_ebl_feedback_accuracy.npy')

Beta_chang_trials = [20, 40, 60, 80]

slow_cng_acc = np.load(slow_cng_dir)
sudden_cng_acc = np.load(sudden_cng_dir)


## Plot suddent change in type of feedbacks
sud_mean_acc = np.mean(sudden_cng_acc,axis=0)
sud_std_acc = np.std(sudden_cng_acc,axis=0)

t = np.arange(1,sud_mean_acc.shape[0]+1,1)

## PLot the bar chart across 3 conditions
label = "$\\beta$-change"
ax2[2].plot(t, sud_mean_acc, alpha=0.75, color='tab:orange')
ax2[2].axvline(x=Beta_chang_trials[0], color='k', linestyle='dashed',lw=1, label=label)
ax2[2].set_ylim([0, 0.85])
ax2[2].spines['right'].set_visible(False)
ax2[2].spines['top'].set_visible(False)
ax2[2].set_ylabel('Motor error')
ax2[2].set_xlabel('Trials (x10)')
ax2[2].xaxis.set_ticks_position('none') 
ax2[2].yaxis.set_ticks_position('none') 
ax2[2].legend(loc='upper left', bbox_to_anchor=(0.2, 1), frameon=False,fontsize=font_s)

## Plot slow change in type of feedbacks
slow_mean_acc = np.mean(slow_cng_acc,axis=0)
slow_std_acc = np.std(slow_cng_acc,axis=0)


## PLot the bar chart across 3 conditions
ax2[3].plot(t, slow_mean_acc, alpha=0.75, color='tab:orange')
for c in Beta_chang_trials:
    ax2[3].axvline(x=c, color='k', linestyle='dashed',lw=1, label=label)
ax2[2].set_ylim([0, 0.85])
ax2[3].spines['right'].set_visible(False)
ax2[3].spines['top'].set_visible(False)
ax2[3].set_xlabel('Trials (x10)')
ax2[3].xaxis.set_ticks_position('none') 
ax2[3].yaxis.set_ticks_position('none') 
ax2[3].set_ylim([0, 0.15])
#ax2[3].set_yticks([])

plt.show()
## SAVE: figure
#plt.savefig('/Users/px19783/Desktop/TwoPolicies_results', format='png', dpi=1400)
