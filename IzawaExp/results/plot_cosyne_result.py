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
fig, ax2 = plt.subplots(nrows=1, ncols=7, figsize=(10,2),
 gridspec_kw={'wspace': 0.65, 'hspace': 0.3, 'left': 0.05, 'right': 0.95, 'bottom': 0.25,
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
    ax2[0].bar(x[i],outcome_variabilities[i], width=width, align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k', color=colors[i], label=conditions[i]) #color='tab:gray',
ax2[0].bar(x+width,human_data, width=width, align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k', color='tab:gray')
ax2[0].errorbar(conditions,outcome_variabilities, yerr=outcome_variabilities_std, ls='none', color='black',  elinewidth=0.75, capsize=1.5) # ecolor='lightslategray',
ax2[0].set_xticks(x + width/2)
ax2[0].set_xticklabels(conditions)
ax2[0].set_ylim([0, 4])
ax2[0].set_yticks([0,1,2,3,4])
ax2[0].spines['right'].set_visible(False)
ax2[0].spines['top'].set_visible(False)
ax2[0].set_ylabel('Reach variability [deg]')
ax2[0].xaxis.set_ticks_position('none') 
ax2[0].yaxis.set_ticks_position('none') 
ax2[0].legend(loc='upper left', bbox_to_anchor=(0, -0.12), frameon=False,fontsize=font_s, ncol=3)
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
ax2[1].bar(x,mean_adapt, width=width, align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k', color=colors) #color='tab:gray',
ax2[1].bar(x+width,human_adaptation, width=width, align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k', color='tab:gray', label='experimental data') #color='tab:gray',
ax2[1].errorbar(x,mean_adapt, yerr=stde_adapt, ls='none', color='black',  elinewidth=0.75, capsize=1.5) # ecolor='lightslategray',
ax2[1].set_xticks(x + width/2)
ax2[1].set_xticklabels(conditions)
ax2[1].set_ylim([0, 10])
ax2[1].spines['right'].set_visible(False)
ax2[1].spines['top'].set_visible(False)
ax2[1].set_ylabel('Adaptation [deg]')
ax2[1].xaxis.set_ticks_position('none') 
ax2[1].yaxis.set_ticks_position('none') 
ax2[1].legend(loc='upper left', bbox_to_anchor=(0.7, -0.12), frameon=False,fontsize=font_s)




##  ========== Plot standard generalisation results =============
# Plot standard generalisation
gen_dir = os.path.join(file_dir,'generalisation')
stand_gen_file = os.path.join(gen_dir,'gener_results.npy')
generalisation_res = np.load(stand_gen_file)
gen_mean = np.round(np.flip(generalisation_res[0,:]),3)
gen_std = np.flip(generalisation_res[1,:]) /normalising_sde

for i, c in enumerate(colors):
    ax2[2].plot(i, gen_mean[i], 'o', markersize=5, color=c, alpha=0.5)
ax2[2].errorbar(conditions, gen_mean, yerr=gen_std,capsize=3, ecolor = "black", elinewidth=0.75, color=c, fmt="None")
ax2[2].spines['right'].set_visible(False)
ax2[2].spines['top'].set_visible(False)
ax2[2].set_ylabel('Generalization error')
ax2[2].xaxis.set_ticks_position('none') 
ax2[2].yaxis.set_ticks_position('none') 

## ======= Plot best betas across Noise level =======
# Best beta across noise
betas_acc_file = os.path.join(gen_dir,'Noise_generalisation_best_betas.npy')
best_betas_data = np.load(betas_acc_file) # [noise_amounts, mean_best_betas, std_best_betas]

betas_noise = best_betas_data[0,:]
mean_betas = best_betas_data[1,:]
std_betas = best_betas_data[2,:] / normalising_sde

ax2[3].errorbar(betas_noise, mean_betas, yerr=std_betas, capsize=3, fmt="r--o", ecolor = "black",markersize=2.5,capthick=0.75,elinewidth=0.75, alpha=0.75, color='tab:orange')
ax2[3].spines['right'].set_visible(False)
ax2[3].spines['top'].set_visible(False)
ax2[3].set_ylabel('Optimal '+"$\\beta$")
ax2[3].set_xlabel('Sensory noise')
ax2[3].xaxis.set_ticks_position('none') 
ax2[3].yaxis.set_ticks_position('none') 



## ============= LINE DRAWING TASK =========
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'../../Drawing_exp/results/')

n_updates = 3
n_seeds = 5
ataxia_mean = []
ataxia_std = []
## Load data for three types of errors across 3 conditions (i.e. update every 1 step, every 2 step, etc..)
for dd in range(1,n_updates+1):
    # Load Ataxia scores
    ataxia_score = np.load(os.path.join(file_dir,str(dd)+'_update','Ataxia_score.npy'))
    ataxia_mean.append(ataxia_score[0,:])
    ataxia_std.append(ataxia_score[1,:])

ataxia_mean = np.flip(np.array(ataxia_mean),axis=1)
ataxia_se = np.flip(np.array(ataxia_std),axis=1) / np.sqrt(n_seeds)

# Load outcomes only for n_update_x_step = 1, i.e. only plot hand traject for this condition
update = 1 

outcome_data = np.load(os.path.join(file_dir,str(update)+'_update','Traject_outcomes.npy'))
mixedModel_norm_grad = np.load(os.path.join(file_dir,str(update)+'_update','MixedModel_grads.npy'))

conditions = ['EBL', 'Mixed', 'RBL']
i=0

ax2[4].bar(conditions,ataxia_mean[0,:],align='center', alpha=0.5,ecolor='black', capsize=5, color=colors,edgecolor=None)
ax2[4].errorbar(conditions,ataxia_mean[0,:], yerr=ataxia_se[0,:], ls='none', color='black',  elinewidth=1, capsize=1.5) # ecolor='lightslategray',
ax2[4].spines['right'].set_visible(False)
ax2[4].spines['top'].set_visible(False)
ax2[4].set_ylabel('Ataxia score')
ax2[4].legend(loc='upper center', bbox_to_anchor=(0.5, 1), frameon=False, fontsize=font_s)#, ncol=5)
ax2[4].xaxis.set_ticks_position('none') 
ax2[4].yaxis.set_ticks_position('none') 


## Plot ataxia changes by temporal sensory feedback
#axs[1,1].errorbar(np.arange(1,n_updates+1),ataxia_mean, yerr=ataxia_std, capsize=3, fmt="r--o", ecolor = "black")
#axs[1,1].plot(np.arange(1,n_updates+1),ataxia_mean, label=conditions)

conditions = np.arange(1,n_updates+1).repeat(3).reshape(3,3)
i=0
for m,s in zip(ataxia_mean,ataxia_se):
    if i ==0:
        ax2[5].errorbar(conditions[i,0],m[0], yerr=s[0], label='EBL', capsize=3, fmt="r--o",markerfacecolor=colors[0],markeredgecolor=colors[0],c='black',elinewidth=1,alpha=0.5)
        ax2[5].errorbar(conditions[i,1],m[1], yerr=s[1], label='Mixed', capsize=3, fmt="r--o",markerfacecolor=colors[1],markeredgecolor=colors[1],c='black',elinewidth=1,alpha=0.5)
        ax2[5].errorbar(conditions[i,2],m[2], yerr=s[2], label='RBL', capsize=3, fmt="r--o",markerfacecolor=colors[2],markeredgecolor=colors[2],c='black',elinewidth=1,alpha=0.5)
    else:
        ax2[5].errorbar(conditions[i,0],m[0], yerr=s[0], capsize=3, fmt="r--o",markerfacecolor=colors[0],markeredgecolor=colors[0],c='black',elinewidth=1,alpha=0.5)
        ax2[5].errorbar(conditions[i,1],m[1], yerr=s[1], capsize=3, fmt="r--o",markerfacecolor=colors[1],markeredgecolor=colors[1],c='black',elinewidth=1,alpha=0.5)
        ax2[5].errorbar(conditions[i,2],m[2], yerr=s[2], capsize=3, fmt="r--o",markerfacecolor=colors[2],markeredgecolor=colors[2],c='black',elinewidth=1,alpha=0.5)
    i+=1

ax2[5].spines['right'].set_visible(False)
ax2[5].spines['top'].set_visible(False)
ax2[5].set_xlabel('sensory feedback delay')
ax2[5].set_ylabel('Accuracy')
#ax2[5].legend(loc='upper center', bbox_to_anchor=(0.35, 1.05), frameon=False, fontsize=font_s)#, ncol=5)
ax2[5].xaxis.set_ticks_position('none') 
ax2[5].yaxis.set_ticks_position('none') 


## Plot training gradients for Mixed condition only for n_update_x_step = 1

RBL_grad = mixedModel_norm_grad[0,:] 
EBL_grad = mixedModel_norm_grad[1,:]

RBL_grad /= np.max(RBL_grad)
EBL_grad /= np.max(EBL_grad)


t = np.arange(1,len(RBL_grad)+1)
ax2[6].plot(t,RBL_grad,label='DA',c=colors[2],alpha=0.5)
ax2[6].plot(t,EBL_grad,label='CB',c=colors[0],alpha=0.6)
ax2[6].spines['right'].set_visible(False)
ax2[6].spines['top'].set_visible(False)
ax2[6].set_ylabel('Activity')
ax2[6].set_xlabel('episode x 100')
ax2[6].legend(loc='upper center', bbox_to_anchor=(0.5, 1), frameon=False, fontsize=font_s)#, ncol=5)
ax2[6].xaxis.set_ticks_position('none') 
ax2[6].yaxis.set_ticks_position('none') 



plt.show()
## SAVE: figure
#plt.savefig('/Users/px19783/Desktop/Cosyn_all', format='png', dpi=1400)
