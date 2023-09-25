import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

file_dir = os.path.dirname(os.path.abspath(__file__))

ataxia_data = []
error_dirs = ['RBL','Mixed','EBL']
n_updates = 3
## Load data for three types of errors across 3 conditions (i.e. update every 1 step, every 2 step, etc..)
for dd in range(1,n_updates+1):
    for d in error_dirs:
        # Load Ataxia scores
        ataxia_score = np.load(os.path.join(file_dir,str(dd)+'_update',d+'_ataxia_score.npy'))
        ataxia_data.append(sum(ataxia_score)/len(ataxia_score))

ataxia_data = np.array(ataxia_data).reshape(n_updates,len(error_dirs))

# Load outcomes only for n_update_x_step = 1
update = 1 

RBL_outcome = np.load(os.path.join(file_dir,str(update)+'_update',error_dirs[0]+'_outcomes.npy'))
Mixed_outcome = np.load(os.path.join(file_dir,str(update)+'_update',error_dirs[1]+'_outcomes.npy'))
EBL_outcome = np.load(os.path.join(file_dir,str(update)+'_update',error_dirs[2]+'_outcomes.npy'))

outcome_data = [RBL_outcome, Mixed_outcome, EBL_outcome]

font_s =7
mpl.rc('font', size=font_s)
plt.rcParams["font.family"] = "helvetica"
mpl.rcParams['xtick.labelsize'] = font_s 
mpl.rcParams['ytick.labelsize'] = font_s 

fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(7.5,3),
 gridspec_kw={'wspace': 0.32, 'hspace': 0.3, 'left': 0.1, 'right': 0.95, 'bottom': 0.15,
                                               'top': 0.9})

conditions = ['RBL', 'Mixed', 'EBL']
i=0
for d in outcome_data:
    axs[i].plot(d[:,:,0],d[:,:,1])
    # plot required adaptation
    #axs[i].plot(t,optimal_adaptation,c='yellow',linewidth=1)
    #axs[i].set_ylim([-10, 20])
    axs[i].set_title(conditions[i],fontsize=font_s)
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)
    axs[i].set_xlabel('x-coord')
    if i == 0:
        axs[i].set_ylabel('y-coord')
    if i > 0:    
        axs[i].set_yticks([])
    i+=1

axs[3].bar(conditions,ataxia_data[0,:],align='center', alpha=0.5,ecolor='black', capsize=5, color='tab:gray',edgecolor='k')
axs[3].spines['right'].set_visible(False)
axs[3].spines['top'].set_visible(False)

plt.show()
#plt.savefig('/Users/px19783/Desktop/ActionGrad_2nd_experiment_writing', format='png', dpi=1200)
