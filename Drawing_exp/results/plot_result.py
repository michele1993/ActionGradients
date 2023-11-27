import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

file_dir = os.path.dirname(os.path.abspath(__file__))

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

ataxia_mean = np.array(ataxia_mean) 
ataxia_se = np.array(ataxia_std) / np.sqrt(n_seeds)

# Load outcomes only for n_update_x_step = 1, i.e. only plot hand traject for this condition
update = 1 

outcome_data = np.load(os.path.join(file_dir,str(update)+'_update','Traject_outcomes.npy'))
mixedModel_norm_grad = np.load(os.path.join(file_dir,str(update)+'_update','MixedModel_grads.npy'))

font_s =7
mpl.rc('font', size=font_s)
plt.rcParams["font.family"] = "helvetica"
mpl.rcParams['xtick.labelsize'] = font_s 
mpl.rcParams['ytick.labelsize'] = font_s 

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(7.5,3),
 gridspec_kw={'wspace': 0.35, 'hspace': 0.4, 'left': 0.07, 'right': 0.98, 'bottom': 0.15,
                                               'top': 0.9})

conditions = ['RBL', 'Mixed', 'EBL']
#colors = ['tab:red','tab:green','tab:blue']
i=0
for d in outcome_data:
    axs[0,i].plot(d[:,:,0],d[:,:,1])
    #axs[0,i].plot(d[:,:,0],d[:,:,1], color=colors[i],alpha=0.5)
    # plot required adaptation
    #axs[i].plot(t,optimal_adaptation,c='yellow',linewidth=1)
    #axs[i].set_ylim([-10, 20])
    axs[0,i].set_title(conditions[i],fontsize=font_s)
    axs[0,i].spines['right'].set_visible(False)
    axs[0,i].spines['top'].set_visible(False)
    axs[0,i].set_xlabel('x-coord')
    if i == 0:
        axs[0,i].set_ylabel('y-coord')
    if i > 0:    
        axs[0,i].set_yticks([])
    i+=1

colors = ['tab:blue','tab:green','tab:orange']
axs[1,0].bar(conditions,ataxia_mean[0,:],align='center', alpha=0.8,ecolor='black', capsize=5, color=colors,edgecolor=None)
axs[1,0].errorbar(conditions,ataxia_mean[0,:], yerr=ataxia_se[0,:], ls='none', color='black',  elinewidth=1, capsize=1.5) # ecolor='lightslategray',
axs[1,0].spines['right'].set_visible(False)
axs[1,0].spines['top'].set_visible(False)
axs[1,0].set_ylabel('Ataxia score')
axs[1,0].legend(loc='upper center', bbox_to_anchor=(0.5, 1), frameon=False, fontsize=font_s)#, ncol=5)


## Plot ataxia changes by temporal sensory feedback
#axs[1,1].errorbar(np.arange(1,n_updates+1),ataxia_mean, yerr=ataxia_std, capsize=3, fmt="r--o", ecolor = "black")
#axs[1,1].plot(np.arange(1,n_updates+1),ataxia_mean, label=conditions)

conditions = np.arange(1,n_updates+1).repeat(3).reshape(3,3)
i=0
for m,s in zip(ataxia_mean,ataxia_se):
    if i ==0:
        axs[1,1].errorbar(conditions[i,0],m[0], yerr=s[0], label='RBL', capsize=3, fmt="r--o",markerfacecolor=colors[0],markeredgecolor=colors[0],c='black',elinewidth=1,alpha=0.8)
        axs[1,1].errorbar(conditions[i,1],m[1], yerr=s[1], label='Mixed', capsize=3, fmt="r--o",markerfacecolor=colors[1],markeredgecolor=colors[1],c='black',elinewidth=1,alpha=0.8)
        axs[1,1].errorbar(conditions[i,2],m[2], yerr=s[2], label='EBL', capsize=3, fmt="r--o",markerfacecolor=colors[2],markeredgecolor=colors[2],c='black',elinewidth=1,alpha=0.8)
    else:
        axs[1,1].errorbar(conditions[i,0],m[0], yerr=s[0], capsize=3, fmt="r--o",markerfacecolor=colors[0],markeredgecolor=colors[0],c='black',elinewidth=1,alpha=0.8)
        axs[1,1].errorbar(conditions[i,1],m[1], yerr=s[1], capsize=3, fmt="r--o",markerfacecolor=colors[1],markeredgecolor=colors[1],c='black',elinewidth=1,alpha=0.8)
        axs[1,1].errorbar(conditions[i,2],m[2], yerr=s[2], capsize=3, fmt="r--o",markerfacecolor=colors[2],markeredgecolor=colors[2],c='black',elinewidth=1,alpha=0.8)
    i+=1

axs[1,1].spines['right'].set_visible(False)
axs[1,1].spines['top'].set_visible(False)
axs[1,1].set_xlabel('sensory temporal feedback')
axs[1,1].set_ylabel('Ataxia score')
axs[1,1].legend(loc='upper center', bbox_to_anchor=(0.25, 1.05), frameon=False, fontsize=font_s)#, ncol=5)


## Plot training gradients for Mixed condition only for n_update_x_step = 1

RBL_grad = mixedModel_norm_grad[0,:]
EBL_grad = mixedModel_norm_grad[1,:]

t = np.arange(1,len(RBL_grad)+1)
axs[1,2].plot(t,RBL_grad,label='RBL',c=colors[0],alpha=0.8)
axs[1,2].plot(t,EBL_grad,label='EBL',c=colors[1],alpha=0.8)
axs[1,2].spines['right'].set_visible(False)
axs[1,2].spines['top'].set_visible(False)
axs[1,2].set_ylabel('Gradient norm')
axs[1,2].set_xlabel('episode x 100')
axs[1,2].legend(loc='upper center', bbox_to_anchor=(0.5, 1), frameon=False, fontsize=font_s)#, ncol=5)

plt.show()
#plt.savefig('/Users/px19783/Desktop/LineDrawingResults', format='png', dpi=1400)
