import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import set_matplotlib_formats
from numpy import genfromtxt


radiant_ratio = 360/(2*np.pi)
## ================= Load data ===================
data_dir = os.path.dirname(os.path.abspath(__file__))
data_1 = 'RBL_results.pt'
data_2 = 'EBL_results.pt'

file_dir = os.path.join(data_dir,data_1)
RBL_data = torch.load(file_dir)

file_dir = os.path.join(data_dir,data_2)
EBL_data = torch.load(file_dir)


# Load general variables shared across all conditions
n_baseline_trl = RBL_data['n_baseline_trl']
n_perturb_trl = RBL_data['n_perturb_trl']
origin = RBL_data['Origin']
targets = RBL_data['Targets']

# Load RBL data
#RBL_xy_acc = RBL_data['XY_accuracy']
#RBL_actions = RBL_data['Actions']
RBL_angle_acc = RBL_data['Angle_accyracy'] * radiant_ratio
RBL_xy_outcomes = RBL_data['Outcomes']
RBL_direct_e = RBL_data['Direct_error']

# Load EBL data
#EBL_xy_acc = EBL_data['XY_accuracy']
#EBL_actions = EBL_data['Actions']
EBL_angle_acc = EBL_data['Angle_accyracy'] * radiant_ratio
EBL_xy_outcomes = EBL_data['Outcomes']
EBL_direct_e = EBL_data['Direct_error']

# Load Mixed data across beta values [0.25, 0.5, 0.75] and store in a list
betas = [0.25, 0.5, 0.75]
Mixed_angle_accuracieS = []
Mixed_xy_outcomeS = []
Mixed_dir_errorS = []
for b in betas:
    data_3 = 'Mixed_'+str(b)+'_results.pt'
    file_dir = os.path.join(data_dir,data_3)
    Mixed_data = torch.load(file_dir)
    Mixed_angle_acc = Mixed_data['Angle_accyracy'] * radiant_ratio
    Mixed_xy_outcomes = Mixed_data['Outcomes']
    Mixed_direct_e = Mixed_data['Direct_error']
    Mixed_angle_accuracieS.append(Mixed_angle_acc)
    Mixed_xy_outcomeS.append(Mixed_xy_outcomes)
    Mixed_dir_errorS.append(Mixed_direct_e)
## ==============================================


font_s = 7
mpl.rc('font', size=font_s)
plt.rcParams["font.family"] = "helvetica"
mpl.rcParams['xtick.labelsize'] = font_s 
mpl.rcParams['ytick.labelsize'] = font_s 

conditions = ['CB-driven', 'CB-ablation']

# Subplot
#fig, ax2 = plt.subplots(nrows=2, ncols=4, figsize=(7,4),
# gridspec_kw={'wspace': 0.65, 'hspace': 0.4, 'left': 0.1, 'right': 0.95, 'bottom': 0.1,
#                                               'top': 0.95})

fig = plt.figure(figsize=(7, 4))
gs = fig.add_gridspec(nrows=2, ncols=4, wspace=0.4, hspace=0.4, left=0.1, right=0.95, bottom=0.1, top=0.95)#, height_ratios=[2,2,1,2])


colors = ['tab:blue','tab:red','tab:green']

final_pert_trial = n_baseline_trl + n_perturb_trl -2
start_washout_trial = n_baseline_trl + n_perturb_trl -1

mean_rbl_angle_error = RBL_angle_acc.mean(axis=0) 
mean_ebl_angle_error = EBL_angle_acc.mean(axis=0)

## Give appropriate sign to trials during washout
## The reson for this is that the angle is computed based on arcos, which only maps [0,pi], excluding negative angles
## However, during the washout phase, the angle is flipped (I checked this)
mean_rbl_angle_error[start_washout_trial:] *= -1
mean_ebl_angle_error[start_washout_trial:] *= -1

std_rbl_angle_error = RBL_angle_acc.std(axis=0) 
std_ebl_angle_error = EBL_angle_acc.std(axis=0)


## ============ Plot angle trajectories ======================
p_trials = [n_baseline_trl, n_baseline_trl+n_perturb_trl]
n_trials = np.arange(0,len(mean_ebl_angle_error))

# EBL:
#ax2[1,0].errorbar(n_trials, mean_ebl_angle_error, yerr=std_ebl_angle_error,capsize=1,elinewidth=0.1, fmt="o", ecolor = "black",markersize=0,color=colors[0],alpha=0.5)
colors_2 = ['cornflowerblue','lightcoral'] 
ax_00 = fig.add_subplot(gs[0, 0])
ax_00.scatter(n_trials, mean_ebl_angle_error,s=2.5, color=colors_2[0], alpha=0.65)
ax_00.fill_between(p_trials, 0, 1,color="0.5",transform=ax_00.get_xaxis_transform(),alpha= 0.3)
ax_00.set_ylim([-30, 30])
ax_00.set_yticks([-30,-20,-10,0,10,20,30])
ax_00.set_ylabel('Error [deg]')
ax_00.spines['right'].set_visible(False)
ax_00.spines['top'].set_visible(False)
ax_00.set_xlabel('Trials')
ax_00.xaxis.set_ticks_position('none') 
ax_00.yaxis.set_ticks_position('none') 

# Human control:
# Load human control data
file_dir = os.path.join(data_dir,"human_control.csv")
human_control = genfromtxt(file_dir, delimiter=',')

ax_01 = fig.add_subplot(gs[0,1])
ax_01.scatter(human_control[:,0], human_control[:,1],s=2.5, color='tab:gray', alpha=0.5)
ax_01.fill_between(p_trials, 0, 1,color="0.5",transform=ax_01.get_xaxis_transform(),alpha= 0.3)
ax_01.set_ylim([-30, 30])
ax_01.set_yticks([-30,-20,-10,0,10,20,30])
#ax2[0,1].set_ylabel('Error [deg]')
ax_01.spines['right'].set_visible(False)
ax_01.spines['top'].set_visible(False)
ax_01.set_xlabel('Trials')
ax_01.xaxis.set_ticks_position('none') 
ax_01.yaxis.set_ticks_position('none') 


# RBL:
ax_02 = fig.add_subplot(gs[0,2])
ax_02.scatter(n_trials, mean_rbl_angle_error,s=2.5, color=colors_2[1], alpha=0.75)
ax_02.fill_between(p_trials, 0, 1,color="0.5",transform=ax_02.get_xaxis_transform(),alpha= 0.3)
ax_02.set_ylim([-30, 30])
ax_02.set_yticks([-30,-20,-10,0,10,20,30])
#ax2[0,2].set_yticks([2,3,4])
ax_02.spines['right'].set_visible(False)
ax_02.spines['top'].set_visible(False)
ax_02.set_xlabel('Trials')
ax_02.xaxis.set_ticks_position('none') 
ax_02.yaxis.set_ticks_position('none') 

# Load human CB(cerebellar) data
file_dir = os.path.join(data_dir,"human_CB.csv")
human_control = genfromtxt(file_dir, delimiter=',')

CB_p_trials = [n_baseline_trl, n_baseline_trl+n_perturb_trl-10] # CB had fewer trials in Tseng et al., 2007
ax_03 = fig.add_subplot(gs[0,3])
ax_03.scatter(human_control[:,0], human_control[:,1],s=2.5, color='tab:gray', alpha=0.5)
ax_03.fill_between(CB_p_trials, 0, 1,color="0.5",transform=ax_03.get_xaxis_transform(),alpha= 0.3)
ax_03.set_ylim([-30, 30])
ax_03.set_yticks([-30,-20,-10,0,10,20,30])
#ax2[0,3].set_ylabel('Error [deg]')
ax_03.spines['right'].set_visible(False)
ax_03.spines['top'].set_visible(False)
ax_03.set_xlabel('Trials')
ax_03.xaxis.set_ticks_position('none') 
ax_03.yaxis.set_ticks_position('none') 

## ============ Plot residual error in degrees ====================
# Human data:
control_human = 8.1 # Tseng et al (2007) Fig.5B
control_sde = 0.7
CB_human = 12.2 # Tseng et al (2007) Fig.5B
CB_sde = 2
human_angle_error = [control_human, CB_human]
human_err_std = [control_sde, CB_sde]

angle_err_mean = [mean_ebl_angle_error[final_pert_trial], mean_rbl_angle_error[final_pert_trial]]
angle_err_std = [std_ebl_angle_error[final_pert_trial], std_rbl_angle_error[final_pert_trial]]


## Following Tseng et al, select the la

x = np.array([0,1])
width=0.4
ax_10 = fig.add_subplot(gs[1,0])
for i in range(2):
    ax_10.bar(x[i], angle_err_mean[i], width=width, align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k', color=colors[i], label=conditions[i]) #color='tab:gray',
ax_10.bar(x+width,human_angle_error, width=width, align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k', color='tab:gray', label='human data')
ax_10.errorbar(conditions[:2], angle_err_mean, yerr=angle_err_std, ls='none', color='black',  elinewidth=0.75, capsize=1.5) # ecolor='lightslategray',
ax_10.errorbar(x+ width, human_angle_error, yerr=human_err_std, ls='none', color='black',  elinewidth=0.75, capsize=1.5) # ecolor='lightslategray',
ax_10.set_xticks(x + width/2)
ax_10.set_xticklabels(conditions[:2])
ax_10.set_ylim([0, 20])
ax_10.set_yticks([0,5,10,15])
ax_10.spines['right'].set_visible(False)
ax_10.spines['top'].set_visible(False)
ax_10.set_ylabel('Residual error [deg]')
ax_10.xaxis.set_ticks_position('none') 
ax_10.yaxis.set_ticks_position('none') 
ax_10.legend(loc='upper left', bbox_to_anchor=(0.25, 1.19), frameon=False,fontsize=font_s, ncol=3)
## ---------------------------------------------------------------------

## =========================== Plot after-effects ========================
# Human data:
control_human = -10 # Tseng et al (2007) Fig.5B
contol_sde= 0.8
CB_human = -4.6 # Tseng et al (2007) Fig.5B
CB_sde = 2.6
human_after_error = [control_human, CB_human]
human_after_sde = [contol_sde, CB_sde]


angle_after_mean = [mean_ebl_angle_error[start_washout_trial], mean_rbl_angle_error[start_washout_trial]]
angle_after_std = [std_ebl_angle_error[start_washout_trial], std_rbl_angle_error[start_washout_trial]] 


## Following Tseng et al, select the la

x = np.array([0,1])
width=0.4

ax_11 = fig.add_subplot(gs[1,1])
for i in range(2):
    ax_11.bar(x[i], angle_after_mean[i], width=width, align='center', alpha=0.5, ecolor='black', capsize=5, edgecolor='k', color=colors[i], label=conditions[i]) #color='tab:gray',
ax_11.bar(x+width,human_after_error, width=width, align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k', color='tab:gray', label='human data')
ax_11.errorbar(conditions[:2], angle_after_mean, yerr=angle_after_std, ls='none', color='black',  elinewidth=0.75, capsize=1.5) # ecolor='lightslategray',
ax_11.errorbar(x+width, human_after_error, yerr=human_after_sde, ls='none', color='black',  elinewidth=0.75, capsize=1.5) # ecolor='lightslategray',
ax_11.set_xticks(x + width)
ax_11.set_xticklabels(conditions[:2])
ax_11.set_ylim([-20, 0])
ax_11.set_yticks([0,-5,-10,-15])
ax_11.spines['right'].set_visible(False)
ax_11.spines['bottom'].set_visible(False)
#ax2[1,1].set_ylabel('Residual error [deg]')
ax_11.xaxis.set_ticks_position('none') 
ax_11.yaxis.set_ticks_position('none') 
#ax2[1,1].tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
#ax2[1,1].spines['top'].set_position(('outward', 10))
#ax2[1,1].xaxis.tick_top()


## ==================== Plot relation between CB contribution (\beta) & adaptation ==================
mean_angle_acc_25 = Mixed_angle_accuracieS[0].mean(axis=0)
mean_angle_acc_50 = Mixed_angle_accuracieS[1].mean(axis=0)
mean_angle_acc_75 = Mixed_angle_accuracieS[2].mean(axis=0)

std_angle_acc_25 = Mixed_angle_accuracieS[0].std(axis=0)
std_angle_acc_50 = Mixed_angle_accuracieS[1].std(axis=0)
std_angle_acc_75 = Mixed_angle_accuracieS[2].std(axis=0)

final_adpt_acc_mean = [mean_rbl_angle_error[final_pert_trial], mean_angle_acc_25[final_pert_trial], mean_angle_acc_50[final_pert_trial],mean_angle_acc_75[final_pert_trial],   mean_ebl_angle_error[final_pert_trial]]
sde_nom = np.sqrt(Mixed_angle_accuracieS[0].shape[0]) # n. of  seeds
final_adpt_acc_std = [std_rbl_angle_error[final_pert_trial]/sde_nom, std_angle_acc_25[final_pert_trial]/sde_nom, std_angle_acc_50[final_pert_trial]/sde_nom,std_angle_acc_75[final_pert_trial]/sde_nom, std_ebl_angle_error[final_pert_trial]/sde_nom]


conditions = [0,25,50,75,100]
condition_labels = ['0%', '25%','50%','75%', '100%']
ax_12 = fig.add_subplot(gs[1,2])
ax_12.errorbar(conditions, final_adpt_acc_mean, yerr=final_adpt_acc_std, capsize=3, fmt="r--o", ecolor = "black",markersize=4,color='tab:orange',alpha=0.5)
ax_12.spines['right'].set_visible(False)
ax_12.spines['top'].set_visible(False)
ax_12.set_ylabel('Residual error [deg]')
ax_12.set_xlabel('CB contribution')
ax_12.set_xticks([0,25,50,75,100])
ax_12.set_xticklabels(condition_labels)
ax_12.xaxis.set_ticks_position('none') 
ax_12.yaxis.set_ticks_position('none') 

## ==================== Plot relation directed errors & change in reaching tarjectories ====================
## Basically: I compute how much each reach endpoint is changing in xy-coord from one trial to the next
## and compute the cosine similarity between that and the corresponding directed error 

# Compute change in reaching trajectory by subtracting next xy end point from current trial end point
RBL_traject_change = torch.tensor(RBL_xy_outcomes[:,1:] - RBL_xy_outcomes[:,0:-1])
EBL_traject_change = torch.tensor(EBL_xy_outcomes[:,1:] - EBL_xy_outcomes[:,0:-1])

# Exclude directed error for last trials - i.e., not used to update action since no more trials left
RBL_direct_e = torch.tensor(RBL_direct_e[:,:-1])
EBL_direct_e = torch.tensor(EBL_direct_e[:,:-1])

# Compute the cosyne similarity
RBL_similarity = torch.nn.functional.cosine_similarity(RBL_traject_change, RBL_direct_e,dim=-1) 
EBL_similarity = torch.nn.functional.cosine_similarity(EBL_traject_change, EBL_direct_e,dim=-1) 

## Compute cosyne similarities for each beta using a loop
Mixed_angle_similaritieS = []
for m in range(len(Mixed_dir_errorS)):
    Mixed_traject_change = torch.tensor(Mixed_xy_outcomeS[m][:,1:] - Mixed_xy_outcomeS[m][:,0:-1])
    Mixed_direct_e = torch.tensor(Mixed_dir_errorS[m][:,:-1])
    Mixed_similarity = torch.nn.functional.cosine_similarity(Mixed_traject_change, Mixed_direct_e,dim=-1) 
    Mixed_angle_similaritieS.append(Mixed_similarity)

## n. of samples: n_seeds x n_targets x n_trials 
sde_norm = np.sqrt(RBL_similarity.size()[0] * RBL_similarity.size()[1] * RBL_similarity.size()[2])

EBL_sim_mean = torch.mean(EBL_similarity)
RBL_sim_mean = torch.mean(RBL_similarity)
Mixed_sim_mean_25 = torch.mean(Mixed_angle_similaritieS[0])
Mixed_sim_mean_50 = torch.mean(Mixed_angle_similaritieS[1])
Mixed_sim_mean_75 = torch.mean(Mixed_angle_similaritieS[2])

mean_sim = [RBL_sim_mean, Mixed_sim_mean_25, Mixed_sim_mean_50, Mixed_sim_mean_75, EBL_sim_mean] 

EBL_sim_sde = torch.std(EBL_similarity)/ sde_norm
RBL_sim_sde = torch.std(RBL_similarity)/ sde_norm
Mixed_sim_sde_25 = torch.std(Mixed_angle_similaritieS[0])/ sde_norm
Mixed_sim_sde_50 = torch.std(Mixed_angle_similaritieS[1])/ sde_norm
Mixed_sim_sde_75 = torch.std(Mixed_angle_similaritieS[2])/ sde_norm

sde_sim = [RBL_sim_sde, Mixed_sim_sde_25, Mixed_sim_sde_50, Mixed_sim_sde_75, EBL_sim_sde] 
conditions = [0,25,50,75,100]
condition_labels = ['0%', '25%','50%','75%', '100%']
ax_13 = fig.add_subplot(gs[1,3])
ax_13.errorbar(conditions, mean_sim, yerr=sde_sim, capsize=3, fmt="r--o", ecolor = "black",markersize=4,color='tab:orange',alpha=0.5)
ax_13.spines['right'].set_visible(False)
ax_13.spines['top'].set_visible(False)
ax_13.set_ylabel('Cosyne similarity')
ax_13.set_xlabel('CB contribution')
ax_13.set_xticks([0,25,50,75,100])
ax_13.set_xticklabels(condition_labels)
ax_13.xaxis.set_ticks_position('none') 
ax_13.yaxis.set_ticks_position('none') 



## ================================= PLOT RESULTS FROM dy/da PERTURBATIONS ======================================
#data_dir = os.path.join(data_dir,'perturbations')
##Load xy coord of reaching 
#mean_xy_outcomeS = []
#file_dir = os.path.join(data_dir,'EBL_NoPerturb_results.pt') # Load non perturbed 'healthy' data
#healty_data = torch.load(file_dir)
#mean_xy_outcomeS.append(healty_data['Outcomes'].mean(axis=0))
#targets = healty_data['Targets']
#x_targ = targets[0,:]
#y_targ = targets[1,:]
## Load data for different dy/da component perturbatios
#label = 'EBL_'
#perturbed_components = [0,1,3]
#for p in perturbed_components:
#    l = label +str(p)+'st_component_results.pt'
#    file_dir = os.path.join(data_dir,l)
#    data = torch.load(file_dir)
#    perturbed_xy_outcomes = data['Outcomes']
#    mean_xy_outcomeS.append(perturbed_xy_outcomes.mean(axis=0))
#i=0
#for o in mean_xy_outcomeS:
#    ax = fig.add_subplot(gs[2,i])
#    sampled_trials = [-1,-2,-3,-4,-5]#np.arange(0,10) * 10
#    x_traj = o[sampled_trials,...,0]#, indx_target_plotted]
#    y_traj = o[sampled_trials,...,1]#, indx_target_plotted]
#    indx_target_plotted = [0,1,2]
#    ax.scatter(x_traj, y_traj, color='b')
#    ax.scatter(x_targ[indx_target_plotted], y_targ[indx_target_plotted], color='r')
#    ax.set_ylim([0.45,0.65])
#    ax.set_xlim([-0.05,0.08])
#    i+=1
## ==========================================================

plt.tight_layout()
plt.show()
