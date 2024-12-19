import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib.patches import Patch

direct = os.path.dirname(os.path.abspath(__file__))
direct = os.path.join(direct,'results')

## Load data with normal DA
DA_reduction = 1

# RBL
offline_RBL_data = np.load(f'Offline_RBL_DA_x{DA_reduction}_reduction_data.npy')
no_offline_RBL_data = np.load(f'No_Offline_RBL_DA_x{DA_reduction}_reduction_data.npy')

# COmpute mean and std for offline and no offline data
mean_off_RBL = offline_RBL_data.mean(axis=0)
std_off_RBL = offline_RBL_data.std(axis=0)

mean_RBL = no_offline_RBL_data.mean(axis=0)
std_RBL = no_offline_RBL_data.std(axis=0)

# EBL
offline_EBL_data = np.load(f'Offline_EBL_DA_x{DA_reduction}_reduction_data.npy')
no_offline_EBL_data = np.load(f'No_Offline_EBL_DA_x{DA_reduction}_reduction_data.npy')

# COmpute mean and std for offline and no offline data
mean_off_EBL = offline_EBL_data.mean(axis=0)
std_off_EBL = offline_EBL_data.std(axis=0)

mean_EBL = no_offline_EBL_data.mean(axis=0)
std_EBL = no_offline_EBL_data.std(axis=0)


font_s =7
mpl.rc('font', size=font_s)
plt.rcParams["font.family"] = "helvetica"
mpl.rcParams['xtick.labelsize'] = font_s 
mpl.rcParams['ytick.labelsize'] = font_s 

conditions = ['EBL','Mixed','RBL']
labels = ['CB-driven\n(model)', 'Mixed\n(model)', 'DA-driven\n(model)']

# Subplot
fig, ax2 = plt.subplots(nrows=1, ncols=5, figsize=(7,3),
 gridspec_kw={'wspace': 0.7, 'hspace': 0, 'left': 0.07, 'right': 0.98, 'bottom': 0.2,
                                               'top': 0.85})

t = np.arange(1, len(mean_RBL)+1,1)
n_online_steps = 25 # x 10
n_offline_steps = len(t) - n_online_steps

# Plot log-term retention for EBL vs RBL for offline and no_offline
final_offline_RBL = mean_off_RBL[-1]
final_RBL = mean_RBL[-1]
final_offline_RBL_std = std_off_RBL[-1]
final_RBL_std = std_RBL[-1]

final_offline_EBL = mean_off_EBL[-1]
final_EBL = mean_EBL[-1]
final_offline_EBL_std = std_off_EBL[-1]
final_EBL_std = std_EBL[-1]


x = np.array([0,1])
mean_values = [final_RBL, final_offline_RBL, final_EBL, final_offline_EBL]
width=0.3
first_row_adjustment = 0.04
labels = ['DA-driven', 'DA-driven+DA-replay', 'CB-driven','CB-driven+DA-replay']
no_offline = [final_RBL, final_EBL]
no_offline_std = [final_RBL_std, final_EBL_std]
offline = [final_offline_RBL, final_offline_EBL]
offline_std = [final_offline_RBL_std, final_offline_EBL_std]


ax2[0].set_position(ax2[0].get_position().translated(0, first_row_adjustment))
ax2[0].bar(x[0], mean_values[0], width=width, align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k',color='tab:red')#,label=labels[0])#, color=colors[0], label=labels[0]) #color='tab:gray',
ax2[0].bar(x[0]+width,mean_values[1], width=width, align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k', hatch='//',color='tab:red',label=labels[1])#, color='tab:gray', label=labels[1])
ax2[0].bar(x[1], mean_values[2], width=width, align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k',color='tab:blue')#,label=labels[2])#, color=colors[1], label=labels[2]) #color='tab:gray',
ax2[0].bar(x[1]+width,mean_values[3], width=width, align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k', hatch='//',color='tab:blue',label=labels[3])#, color='k', label=labels[3])
ax2[0].set_ylabel('Long-term accuracy')
ax2[0].errorbar(x, no_offline, yerr=no_offline_std, ls='none', color='black',  elinewidth=0.75, capsize=1.5) # ecolor='lightslategray',
ax2[0].errorbar(x+width, offline, yerr=offline_std, ls='none', color='black',  elinewidth=0.75, capsize=1.5) # ecolor='lightslategray',
ax2[0].spines['right'].set_visible(False)
ax2[0].spines['top'].set_visible(False)
ax2[0].legend(loc='lower left', bbox_to_anchor=(-0.5, -0.35), frameon=False,fontsize=font_s, ncol=1)
ax2[0].set_xticks(x + width/2)
ax2[0].set_xticklabels(['DA-driven','CB-driven'])
ax2[0].set_title("DA-replay drives\nlong-term performance", fontsize=font_s)
ax2[0].xaxis.set_ticks_position('none')

# Plot RBL offline vs no offline performances
offline_steps = np.arange(n_online_steps, n_online_steps+n_offline_steps,1)
ax2[1].plot(t, mean_off_RBL, color='tab:red', alpha=1)
ax2[1].axvline(n_online_steps, ls='--', lw=0.75,color='k', label="end-of-task")
ax2[1].fill_between(offline_steps, 0,0.8, color='tab:gray',alpha=0.25, label='retention period')
ax2[1].plot(t, mean_RBL, color='tab:red', alpha=0.5)#, label='No-replay')
ax2[1].spines['right'].set_visible(False)
ax2[1].spines['top'].set_visible(False)
ax2[1].set_title("DA-replay consolidates\n DA-driven learning", fontsize=font_s, pad=10)
ax2[1].set_ylabel("Accuracy")
ax2[1].set_xlabel("Steps")
ax2[1].legend(loc='lower left', bbox_to_anchor=(-0.5, -0.3), frameon=False,fontsize=font_s, ncol=3)
ax2[1].set_ylim([0,0.7])
ax2[1].set_xlim([0,50])
ax2[1].set_xticks([0,25])

# Plot EBL offline vs no offline performances
pa1 = Patch(facecolor='tab:red',alpha=1) # edgecolor='black')
pa2 = Patch(facecolor='tab:blue',alpha=1) # edgecolor='black')
pa3 = Patch(facecolor='tab:red',alpha=0.5) # edgecolor='black')
pa4 = Patch(facecolor='tab:blue',alpha=0.5) # edgecolor='black')
ax2[2].plot(t, mean_off_EBL, color='tab:blue', alpha=1)
ax2[2].plot(t, mean_EBL, color='tab:blue', alpha=0.5)#, label='No-replay')
ax2[2].axvline(n_online_steps, ls='--', lw=0.75,color='k')
ax2[2].fill_between(offline_steps, 0,0.8, color='tab:gray',alpha=0.25)
ax2[2].spines['right'].set_visible(False)
ax2[2].spines['top'].set_visible(False)
#ax2[2].set_title("CB-driven learning with\nDA-driven replay", fontsize=font_s, pad=10)
ax2[2].set_title("DA-replay consolidates\nCB-driven learning", fontsize=font_s, pad=10)
ax2[2].set_xlabel("Steps")
ax2[2].set_ylim([0,0.7])
ax2[2].set_xlim([0,50])
ax2[2].set_xticks([0,25])
custom_legend = [Patch(facecolor='blue', edgecolor='red', alpha=0.5,label='No-replay')]

#ax2[2].legend(handles=(pa1,pa2),labels=('No DA-replay','No DA-replay'), ncol=1, loc='lower left', bbox_to_anchor=(1, -0.33), frameon=False,fontsize=font_s)
ax2[2].legend(handles=[pa3,pa4],labels=['','No DA-replay'], ncol=3, handletextpad=0.5, handlelength=1.5, columnspacing=-0.3, loc='lower left', bbox_to_anchor=(0.3, -0.3), frameon=False,fontsize=font_s)
#ax2[2].legend(handles=[pa1,pa2,pa3,pa4],labels=['','DA-replay','','No DA-replay'], ncol=4, handletextpad=0.5, handlelength=1, columnspacing=0.5, loc='lower left', bbox_to_anchor=(0.3, -0.3), frameon=False,fontsize=font_s)
#ax2[2].legend(handles=custom_legend,loc='lower left', bbox_to_anchor=(0.8, -0.33), frameon=False,fontsize=font_s)
              


# Plot long-term performance for DA deficiencies
DA_reductions = [1, 10, 100, 1000]
final_da_means = []
final_da_stds = []
da_means = []
da_stds = []
for DA_reduction in DA_reductions[:-1]: # only plot for x1, x10, x100 DA reduction to keep plots cleaner
    data = np.load(f'Offline_EBL_DA_x{DA_reduction}_reduction_data.npy')
    # Extract mean and std for final performance
    mean_da_reduction = data.mean(axis=0)
    std_da_reduction = data.std(axis=0) / np.sqrt(5)
    # Store entire trajectory
    da_means.append(mean_da_reduction)
    da_stds.append(std_da_reduction)
    # Store only final accuracy
    final_da_means.append(mean_da_reduction[-1])
    final_da_stds.append(std_da_reduction[-1])

# Plot long-term retention trajectories for different DA deficiencies

# Compute memory retention in term of accuracy on last learning trial minus offline acc for trajectory
da_means = np.array(da_means)  
final_online_acc = da_means[:,n_online_steps]
da_retention = np.abs(da_means[:,n_online_steps:] - final_online_acc[:,np.newaxis])
colors = ['tab:red', 'orangered', 'salmon', 'chocolate']
i=0
labels = ['healthy', 'x10', 'x100']#, 'x1000']
for da,da_std in zip(da_retention, da_stds):
    ax2[3].plot(t[:n_offline_steps], da, color=colors[i],alpha=0.9,label=labels[i])
    #ax2[3].fill_between(t[:n_offline_steps], da[n_online_steps:]-da_std[n_online_steps:], da[n_online_steps:]+da_std[n_online_steps:], color=colors[i],alpha=0.25)
    i+=1

ax2[3].set_title('DA deficits lead to\nmemory dacay', fontsize=font_s,pad=10)
ax2[3].set_ylabel('Performance decay')
ax2[3].legend(loc='lower left', bbox_to_anchor=(0.0, 0.7), frameon=False,fontsize=font_s, ncol=1, title='DA reduction:')
ax2[3].spines['right'].set_visible(False)
ax2[3].spines['top'].set_visible(False)
ax2[3].set_xticks([])
ax2[3].set_xlabel('Time after learning')

# Plot final long-term retention for different DA deficiencies
condition_labels = ['healthy', 'x10', 'x100']
conditions = [1,2,3]
ax2[4].errorbar(conditions, final_da_means, yerr=final_da_stds,capsize=3, fmt="r--o", ecolor = "black",markersize=4,color='tab:orange',alpha=0.5)
#ax2[3].set_ylim([1.5, 4])
#ax2[3].set_yticks([2,3,4])
ax2[4].spines['right'].set_visible(False)
ax2[4].spines['top'].set_visible(False)
#ax2[3].set_xlabel('$\\beta$ values')
ax2[4].set_title('DA deficits impair \nlong-term performance',fontsize=font_s,pad=10)
ax2[4].set_xlabel('DA reduction')
ax2[4].set_ylabel('Long-term accuracy')
#ax2[3].xaxis.set_ticks_position('none') 
ax2[4].set_xticks([1,2,3])
ax2[4].set_xticklabels(condition_labels)

plt.show()
