import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


radiant_ratio = 360/(2*np.pi)
fig = plt.figure(figsize=(7, 3))
#gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[1,1])
gs = fig.add_gridspec(nrows=2, ncols=4, wspace=0.4, hspace=0.2, left=0.1, right=0.95, bottom=0.1, top=0.95, height_ratios=[1,1])

font_s = 7
mpl.rc('font', size=font_s)
plt.rcParams["font.family"] = "helvetica"
mpl.rcParams['xtick.labelsize'] = font_s 
mpl.rcParams['ytick.labelsize'] = font_s 

## ================================= PLOT RESULTS FROM dy/da PERTURBATIONS ======================================
data_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(data_dir,'perturbations')

#Load xy coord of reaching 
mean_xy_outcomeS = []
file_dir = os.path.join(data_dir,'EBL_NoPerturb_results.pt') # Load non perturbed 'healthy' data
healty_data = torch.load(file_dir)
mean_xy_outcomeS.append(healty_data['Outcomes'].mean(axis=0))
targets = healty_data['Targets']
x_targ = targets[0,:]
y_targ = targets[1,:]
# Load data for different dy/da component perturbatios
label = 'EBL_'
perturbed_components = [0,1,3]
for p in perturbed_components:
    l = label +str(p)+'st_component_results.pt'
    file_dir = os.path.join(data_dir,l)
    data = torch.load(file_dir)
    perturbed_xy_outcomes = data['Outcomes']
    mean_xy_outcomeS.append(perturbed_xy_outcomes.mean(axis=0))

i=0
e=0
for o in mean_xy_outcomeS:
    if i <2:
        ax = fig.add_subplot(gs[0,i])
        i+=1
    else:
        ax = fig.add_subplot(gs[1,e])
        e+=1
    sampled_trials = np.arange(0,10) * (-1)
    x_traj = o[sampled_trials,...,0]#, indx_target_plotted]
    y_traj = o[sampled_trials,...,1]#, indx_target_plotted]
    plot_target_indx = 1
    # Plot endpoints for each target
    ax.scatter(x_traj[:,plot_target_indx], y_traj[:,plot_target_indx], color='tab:blue', s=15, alpha=0.65, marker='x', label='reaching outcomes')
    #ax.scatter(x_traj[:,0], y_traj[:,0], color='tab:brown', s=12, alpha=0.65, marker='.')
    #ax.scatter(x_traj[:,1], y_traj[:,1], color='tab:blue', s=15, alpha=0.65, marker='x')
    #ax.scatter(x_traj[:,2], y_traj[:,2], color='tab:purple', s=12, alpha=0.65, marker='.')
    # Plot targets
    ax.scatter(x_targ[plot_target_indx], y_targ[plot_target_indx], color='tab:red', s=15, alpha=0.65, marker='*', label='target')
    #ax.scatter(x_targ[0], y_targ[0], color='tab:brown', s=12, alpha=0.65, marker='*')
    #ax.scatter(x_targ[1], y_targ[1], color='tab:red', s=15, alpha=0.65, marker='*')
    #ax.scatter(x_targ[2], y_targ[2], color='tab:purple', s=12, alpha=0.65, marker='*')
    ax.set_ylim([0.45,0.65])
    ax.set_xlim([-0.05,0.09])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    if i+e >3:
        ax.legend(loc='lower left', bbox_to_anchor=(-0.7, -0.3), frameon=False,fontsize=font_s, ncol=2)

## ==========================================================

## Load data across different betas
file_dir = os.path.join(data_dir,'RBL_random_results.pt') # Load non perturbed 'healthy' data
data = torch.load(file_dir) 
RBL_angle_acc = data['Angle_accyracy']
RBL_xy_outcome = data['Outcomes']
targets = np.expand_dims(data['Targets'].T,axis=0)

file_dir = os.path.join(data_dir,'Mixed_0.25_random_results.pt') # Load non perturbed 'healthy' data
data = torch.load(file_dir) 
Mix25_angle_acc = data['Angle_accyracy']
Mix25_xy_outcome = data['Outcomes']

file_dir = os.path.join(data_dir,'Mixed_0.5_random_results.pt') # Load non perturbed 'healthy' data
data = torch.load(file_dir) 
Mix50_angle_acc = data['Angle_accyracy']
Mix50_xy_outcome = data['Outcomes']

file_dir = os.path.join(data_dir,'Mixed_0.75_random_results.pt') # Load non perturbed 'healthy' data
data = torch.load(file_dir) 
Mix75_angle_acc = data['Angle_accyracy']
Mix75_xy_outcome = data['Outcomes']

file_dir = os.path.join(data_dir,'EBL_random_results.pt') # Load non perturbed 'healthy' data
data = torch.load(file_dir) 
EBL_angle_acc = data['Angle_accyracy']
EBL_xy_outcome = data['Outcomes']

## ================= Plot dysmetria scores =======================
outcomes = [RBL_xy_outcome, Mix25_xy_outcome,Mix50_xy_outcome,Mix75_xy_outcome,EBL_xy_outcome]
# Add bottom right subplot - gs[bottom row, last column (the 'left' subplot)]
# We do not add the upper right subplot
dysmetria_mean_score = []
dysmetria_std_score = []
for a in outcomes:
    # NOTE: dysmetria seems to refer to over-shooting (hypermetria) and under-shooting (hypometria)
    # So compute score only relative to the displacement in x-coords
    x_displacement = np.sqrt((targets[...,0] - a[...,0])**2).mean(axis=-1) # mean across 3 targets
    # Compute mean and std across seeds and last n. trials (i.e., sampled_trials)
    mean_x_displ = x_displacement[:,sampled_trials].mean()
    std_x_displ = x_displacement[:,sampled_trials].std()
    #mean_y_displacement = ((targets[...,1] - a[...,1])**2).mean(axis=0).mean(axis=-1)
    dysmetria_mean_score.append(mean_x_displ)
    dysmetria_std_score.append(std_x_displ)

print(dysmetria_mean_score)
print(dysmetria_std_score)
exit()
ax = fig.add_subplot(gs[:,2])
conditions = [0,25,50,75,100]
condition_labels = ['0%', '25%','50%','75%', '100%']
ax.errorbar(conditions, dysmetria_mean_score, yerr=dysmetria_std_score, capsize=3, fmt="r--o", ecolor = "black",markersize=4,color='tab:orange',alpha=0.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Dysmetria score')
ax.set_xlabel('CB contribution')
ax.set_xticks([0,25,50,75,100])
ax.set_xticklabels(condition_labels)
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none') 


## ======= Plot angle accuracy =============
angle_acc = [RBL_angle_acc, Mix25_angle_acc,Mix50_angle_acc,Mix75_angle_acc,EBL_angle_acc]
ax_right_bottom = fig.add_subplot(gs[:, 3])
ax_right_bottom.set_xlabel("Right Bottom X label")
ax_right_bottom.set_ylabel("Right Bottom Y label")

plt.tight_layout()
plt.show()
