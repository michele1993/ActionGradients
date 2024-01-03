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
ax2[0,2].xaxis.set_ticks_position('none') 
ax2[0,2].yaxis.set_ticks_position('none') 

## ======== Plot action nosie adaptation for negative and positive RPE ==========

# Load data for negative RPEs
std_a_dir_NegRPE = os.path.join(file_dir,'beta_grid','Mixed_0_std_a_adaptation_NegRPE.npy')
std_a_NegRPE = np.load(std_a_dir_NegRPE) / 0.0176
mean_std_a_NegRPE = std_a_NegRPE.mean(axis=0)
std_std_a_NegRPE = std_a_NegRPE.std(axis=0) / normalising_sde

# Load data for positive RPEs
std_a_dir_PosRPE = os.path.join(file_dir,'beta_grid','Mixed_0_std_a_adaptation_PosRPE.npy')
std_a_PosRPE = np.load(std_a_dir_PosRPE) / 0.0176
mean_std_a_PosRPE = std_a_PosRPE.mean(axis=0)
std_std_a_PosRPE = std_a_PosRPE.std(axis=0) / normalising_sde

t = np.arange(0,11,1) *10
RPE_colors = ['tab:purple', 'tab:olive']
ax2[0,3].errorbar(t, mean_std_a_NegRPE, yerr=std_std_a_NegRPE, capsize=3, fmt="r--o", ecolor = "black",markersize=4, color=RPE_colors[0], alpha=0.5, label='RPEs < 0')
ax2[0,3].errorbar(t, mean_std_a_PosRPE, yerr=std_std_a_PosRPE, capsize=3, fmt="r--o", ecolor = "black",markersize=4, color=RPE_colors[1], alpha=0.5, label='RPEs > 0')
ax2[0,3].spines['right'].set_visible(False)
ax2[0,3].spines['top'].set_visible(False)
#ax2[0,3].set_ylabel('Action noise')
ax2[0,3].set_xlabel('Trials')
ax2[0,3].xaxis.set_ticks_position('none') 
ax2[0,3].yaxis.set_ticks_position('none') 
ax2[0,3].legend(loc='upper left', bbox_to_anchor=(0.28, 0.64), frameon=False,fontsize=font_s)


##  ========== Plot standard generalisation results =============
# Plot standard generalisation
gen_dir = os.path.join(file_dir,'generalisation')
stand_gen_file = os.path.join(gen_dir,'gener_results.npy')
generalisation_res = np.load(stand_gen_file)
gen_mean = np.round(np.flip(generalisation_res[0,:]),3)
gen_std = np.flip(generalisation_res[1,:]) /normalising_sde

for i, c in enumerate(colors):
    ax2[1,0].plot(i, gen_mean[i], 'o', markersize=5, color=c, alpha=0.5)
ax2[1,0].errorbar(conditions, gen_mean, yerr=gen_std,capsize=3, ecolor = "black", elinewidth=0.75, color=c, fmt="None",alpha=0.5)
ax2[1,0].spines['right'].set_visible(False)
ax2[1,0].spines['top'].set_visible(False)
ax2[1,0].set_ylabel('Generalization error')
ax2[1,0].xaxis.set_ticks_position('none') 
ax2[1,0].yaxis.set_ticks_position('none') 

## ======== Plot training accuracy for policy generalisation =====
gen_trainign_dir = os.path.join(gen_dir,'pol_train_acc.npy')
gen_trainign_acc = np.load(gen_trainign_dir)
n_trials = gen_trainign_acc.shape[-1]
t = np.arange(1,n_trials+1,1)

for i in range(3):
    ax2[1,1].plot(t, gen_trainign_acc[i,:], color=np.flip(colors)[i], alpha=0.5)

ax2[1,1].spines['right'].set_visible(False)
ax2[1,1].spines['top'].set_visible(False)
ax2[1,1].set_ylabel('Training error')
ax2[1,1].set_xlabel('Trials (x100)')
ax2[1,1].xaxis.set_ticks_position('none') 
ax2[1,1].yaxis.set_ticks_position('none') 

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
ax2[1,2].errorbar(sensory_noises, mean_acc, yerr=std_acc, capsize=3, fmt="r--o", ecolor = "black",markersize=4, color=colors[0], alpha=0.5)
ax2[1,2].spines['right'].set_visible(False)
ax2[1,2].spines['top'].set_visible(False)
ax2[1,2].set_ylabel('EBL error')
ax2[1,2].set_xlabel('Sensory noise')
ax2[1,2].xaxis.set_ticks_position('none') 
ax2[1,2].yaxis.set_ticks_position('none') 

# Gradient Error
ax2[1,3].errorbar(sensory_noises, mean_grad_acc, yerr=std_grad_acc, capsize=3, fmt="r--o", ecolor = "black", color=colors[0],markersize=4, alpha=0.5)
ax2[1,3].spines['right'].set_visible(False)
ax2[1,3].spines['top'].set_visible(False)
ax2[1,3].set_ylabel('EBL gradient error')
ax2[1,3].set_xlabel('Sensory noise')
ax2[1,3].xaxis.set_ticks_position('none') 
ax2[1,3].yaxis.set_ticks_position('none') 



plt.show()
## SAVE: figure
#plt.savefig('/Users/px19783/Desktop/Izawa_results', format='png', dpi=1400)
