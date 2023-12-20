import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import set_matplotlib_formats

## =============== Plot motor variability results ===============
## Load data
seeds = [8721, 5467, 1092, 9372,2801]
betas = np.arange(0,11,1) /10.0
normalising_sde = np.sqrt(len(seeds))
file_dir = os.path.dirname(os.path.abspath(__file__))
tot_var = []
for s in seeds:
    acc_dir = os.path.join(file_dir,'beta_grid',str(s))
    per_s_var = []
    for b in betas:
        label = "Mixed_"+str(b)
        outcome_dir = os.path.join(acc_dir,label+'_outcome_variability.npy') # For the mixed model
        var = np.load(outcome_dir)
        per_s_var.append(var)
    tot_var.append(np.array(per_s_var))
 
tot_var = np.array(tot_var) / 0.0176 # convert to degrees
mean_var = np.mean(tot_var, axis=0)
std_var = np.std(tot_var, axis=0) / normalising_sde

font_s =7
mpl.rc('font', size=font_s)
plt.rcParams["font.family"] = "helvetica"
mpl.rcParams['xtick.labelsize'] = font_s 
mpl.rcParams['ytick.labelsize'] = font_s 

conditions = ['EBL','Mixed','RBL']

# Subplot
fig, ax2 = plt.subplots(nrows=2, ncols=4, figsize=(7,4),
 gridspec_kw={'wspace': 0.65, 'hspace': 0.4, 'left': 0.1, 'right': 0.95, 'bottom': 0.1,
                                               'top': 0.95})
EBL_mean_variability = mean_var[-1]
EBL_std_variability = std_var[-1] / normalising_sde

## Based on Izawa's results, use beta=0.4 
Mixed_mean_variability = mean_var[4]
Mixed_std_variability = std_var[4] / normalising_sde

RBL_mean_variability = mean_var[0]
RBL_std_variability = std_var[0] / normalising_sde

outcome_variabilities = [EBL_mean_variability, Mixed_mean_variability, RBL_mean_variability]
outcome_variabilities_std = [EBL_std_variability, Mixed_std_variability, RBL_std_variability]

colors = ['tab:blue','tab:green','tab:red']

human_data = [1.6, 2, 3.5]
x = np.array([0,1,2])
width=0.4
## PLot the bar chart across 3 conditions
for i in range(3):
    ax2[0,1].bar(x[i],outcome_variabilities[i], width=width, align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k', color=colors[i], label=conditions[i]) #color='tab:gray',
ax2[0,1].bar(x+width,human_data, width=width, align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k', color='tab:gray')
ax2[0,1].errorbar(conditions,outcome_variabilities, yerr=outcome_variabilities_std, ls='none', color='black',  elinewidth=0.75, capsize=1.5) # ecolor='lightslategray',
ax2[0,1].set_xticks(x + width/2)
ax2[0,1].set_xticklabels(conditions)
ax2[0,1].set_ylim([0, 4])
ax2[0,1].set_yticks([0,1,2,3,4])
ax2[0,1].spines['right'].set_visible(False)
ax2[0,1].spines['top'].set_visible(False)
ax2[0,1].set_ylabel('Reach variability [deg]')
ax2[0,1].xaxis.set_ticks_position('none') 
ax2[0,1].yaxis.set_ticks_position('none') 
ax2[0,1].legend(loc='upper left', bbox_to_anchor=(-0.7, -0.12), frameon=False,fontsize=font_s, ncol=3)
#ax2[5].legend(loc='upper center', bbox_to_anchor=(0.35, 1.05), frameon=False, fontsize=font_s)#, ncol=5)


# Plot adaptation comparison between humans and the model

human_adaptation = [7.63, 7.55, 7.63]
tot_var = []
for s in seeds:
    acc_dir = os.path.join(file_dir,'beta_grid',str(s))
    per_s_var = []
    for b in betas:
        label = "Mixed_"+str(b)
        outcome_dir = os.path.join(acc_dir,label+'_trajectories.npy') # For the mixed model
        var = np.load(outcome_dir)
        per_s_var.append(var)
    tot_var.append(np.array(per_s_var))
 

tot_var = np.array(tot_var) / 0.0176 # convert to degrees
# Select trajectories for beta=0,0.3,1 (i.e., base on variance analysis resutls, best capturing Izawa's)
results = tot_var.mean(axis=0)[[0,5,10],:].squeeze()
final_trials = results[:,-100:]
mean_adapt = np.flip(final_trials.mean(-1))
stde_adapt = np.flip(final_trials.std(-1) / normalising_sde)

x = np.array([0,1,2])
width=0.4

## PLot the bar chart across 3 conditions
ax2[0,0].bar(x,mean_adapt, width=width, align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k', color=colors) #color='tab:gray',
ax2[0,0].bar(x+width,human_adaptation, width=width, align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k', color='tab:gray', label='experimental data') #color='tab:gray',
ax2[0,0].errorbar(x,mean_adapt, yerr=stde_adapt, ls='none', color='black',  elinewidth=0.75, capsize=1.5) # ecolor='lightslategray',
ax2[0,0].set_xticks(x + width/2)
ax2[0,0].set_xticklabels(conditions)
ax2[0,0].set_ylim([0, 10])
ax2[0,0].spines['right'].set_visible(False)
ax2[0,0].spines['top'].set_visible(False)
ax2[0,0].set_ylabel('Adaptation [deg]')
ax2[0,0].xaxis.set_ticks_position('none') 
ax2[0,0].yaxis.set_ticks_position('none') 
ax2[0,0].legend(loc='upper left', bbox_to_anchor=(-0.2, -0.12), frameon=False,fontsize=font_s)

# Plot 'continous' change in variability across beta values
ax2[0,2].errorbar(betas, mean_var, yerr=std_var,capsize=3, fmt="r--o", ecolor = "black",markersize=4,color='tab:orange',alpha=0.5)
ax2[0,2].set_ylim([1.5, 4])
ax2[0,2].set_yticks([2,3,4])
ax2[0,2].spines['right'].set_visible(False)
ax2[0,2].spines['top'].set_visible(False)
ax2[0,2].set_xlabel('Beta values')




##  ========== Plot standard generalisation results =============
# Plot standard generalisation
gen_dir = os.path.join(file_dir,'generalisation')
stand_gen_file = os.path.join(gen_dir,'gener_results.npy')
generalisation_res = np.load(stand_gen_file)
gen_mean = np.round(np.flip(generalisation_res[0,:]),3)
gen_std = np.flip(generalisation_res[1,:]) /normalising_sde

for i, c in enumerate(colors):
    ax2[0,3].plot(i, gen_mean[i], 'o', markersize=5, color=c, alpha=0.5)
ax2[0,3].errorbar(conditions, gen_mean, yerr=gen_std,capsize=3, ecolor = "black", elinewidth=0.75, color=c, fmt="None")
ax2[0,3].spines['right'].set_visible(False)
ax2[0,3].spines['top'].set_visible(False)
ax2[0,3].set_ylabel('Generalization error')
ax2[0,3].xaxis.set_ticks_position('none') 
ax2[0,3].yaxis.set_ticks_position('none') 

## ======== Plot changes in various accuracies across noise levels =======
## i.e., error, forward model accuracy and gradient accuracy
noise_gen_file = os.path.join(gen_dir,'Noise_generalisation_statistics.npy')
generalisation_res = np.load(noise_gen_file) # [test_seed_acc, test_seed_forward_acc, test_seed_gradSim]

mean_acc = np.mean(generalisation_res[0],axis=0)
mean_frwd_acc = np.mean(generalisation_res[1],axis=0)
mean_grad_acc = np.mean(generalisation_res[2],axis=0)

# cover grad accuracy to grad error for consistency
mean_grad_acc = np.abs(1-mean_grad_acc)

std_acc = np.std(generalisation_res[0],axis=0) / normalising_sde
std_frwd_acc = np.std(generalisation_res[1],axis=0) / normalising_sde
std_grad_acc = np.std(generalisation_res[2],axis=0) / normalising_sde

sensory_noises = torch.linspace(0.01,0.25,10)

# Generalisation Error
ax2[1,0].errorbar(sensory_noises, mean_acc, yerr=std_acc, capsize=3, fmt="r--o", ecolor = "black",markersize=4, color=colors[0], alpha=0.5)
ax2[1,0].spines['right'].set_visible(False)
ax2[1,0].spines['top'].set_visible(False)
ax2[1,0].set_ylabel('EBL error')
ax2[1,0].set_xlabel('Sensory noise')

# Gradient Error
ax2[1,1].errorbar(sensory_noises, mean_grad_acc, yerr=std_grad_acc, capsize=3, fmt="r--o", ecolor = "black", color=colors[0],markersize=4, alpha=0.5)
ax2[1,1].spines['right'].set_visible(False)
ax2[1,1].spines['top'].set_visible(False)
ax2[1,1].set_ylabel('EBL gradient error')
ax2[1,1].set_xlabel('Sensory noise')



plt.show()
## SAVE: figure
#plt.savefig('/Users/px19783/Desktop/Cosyn_all', format='png', dpi=1400)
