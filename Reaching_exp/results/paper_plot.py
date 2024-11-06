import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import genfromtxt
from scipy.stats import norm

save_file = False
radiant_ratio = 360/(2*np.pi)
fig = plt.figure(figsize=(7, 7))
#gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[1,1])
gs = fig.add_gridspec(nrows=4, ncols=4, wspace=0.6, hspace=0.6, left=0.075, right=0.95, bottom=0.05, top=0.93)#, height_ratios=[1,0.2,1])

font_s = 7
mpl.rc('font', size=font_s)
plt.rcParams["font.family"] = "helvetica"
mpl.rcParams['xtick.labelsize'] = font_s 
mpl.rcParams['ytick.labelsize'] = font_s 

root_dir = os.path.dirname(os.path.abspath(__file__))

## =============================== PLOT RESULTS FROM ROTATION TASK ======================

# ------- Load data
data_1 = 'RBL_results.pt'
data_2 = 'EBL_results.pt'
file_dir = os.path.join(root_dir,data_1)
RBL_data = torch.load(file_dir)
file_dir = os.path.join(root_dir,data_2)
EBL_data = torch.load(file_dir)


# Load general variables shared across all conditions
n_baseline_trl = RBL_data['n_baseline_trl']
n_perturb_trl = RBL_data['n_perturb_trl']
origin = RBL_data['Origin']
targets = RBL_data['Targets']

# Load RBL data
RBL_angle_acc = RBL_data['Angle_accyracy'] * radiant_ratio
RBL_xy_outcomes = RBL_data['Outcomes']
RBL_direct_e = RBL_data['Direct_error']

# Load EBL data
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
    file_dir = os.path.join(root_dir,data_3)
    Mixed_data = torch.load(file_dir)
    Mixed_angle_acc = Mixed_data['Angle_accyracy'] * radiant_ratio
    Mixed_xy_outcomes = Mixed_data['Outcomes']
    Mixed_direct_e = Mixed_data['Direct_error']
    Mixed_angle_accuracieS.append(Mixed_angle_acc)
    Mixed_xy_outcomeS.append(Mixed_xy_outcomes)
    Mixed_dir_errorS.append(Mixed_direct_e)

# ------------

# define useful variables
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

## -------------Plot residual error in degrees for rotation task--------------------
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
first_row_adjustment = 0.04

#conditions = ['Control \n (CB driven)', 'CB patients \n (CB-ablation)']
conditions = ['Control', 'CB patients']
labels = ['CB-driven\n(model)', 'human control', 'DA-driven\n(model)', 'CB patients']
x = np.array([0,1])
width=0.3
ax_1 = fig.add_subplot(gs[0,0])
# Increase spacing
ax_1.set_position(ax_1.get_position().translated(0, first_row_adjustment))
#for i in range(2):
ax_1.bar(x[0], angle_err_mean[0], width=width, align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k', color=colors[0], label=labels[0]) #color='tab:gray',
ax_1.bar(x[1], angle_err_mean[1], width=width, align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k', color=colors[1], label=labels[2]) #color='tab:gray',
ax_1.bar(x[0]+width,human_angle_error[0], width=width, align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k', color='tab:gray', label=labels[1])
ax_1.bar(x[1]+width,human_angle_error[1], width=width, align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k', color='k', label=labels[3])
ax_1.errorbar(conditions[:2], angle_err_mean, yerr=angle_err_std, ls='none', color='black',  elinewidth=0.75, capsize=1.5) # ecolor='lightslategray',
ax_1.errorbar(x+ width, human_angle_error, yerr=human_err_std, ls='none', color='black',  elinewidth=0.75, capsize=1.5) # ecolor='lightslategray',
ax_1.set_xticks(x + width/2)
ax_1.set_xticklabels(conditions[:2])
ax_1.set_ylim([0, 20])
ax_1.set_yticks([0,5,10,15])
ax_1.spines['right'].set_visible(False)
ax_1.spines['top'].set_visible(False)
ax_1.set_ylabel('Residual error [deg]')
ax_1.set_title('Rotation task: model vs data', fontsize=font_s)
ax_1.xaxis.set_ticks_position('none') 
#ax_1.yaxis.set_ticks_position('none') 
legend_height = -0.4
ax_1.legend(loc='upper left', bbox_to_anchor=(-0.4, legend_height), frameon=False,fontsize=font_s, ncol=4)
## ---------------------------------------------------------------------

## -------------------- Plot relation between CB contribution (\beta) & adaptation for rotation task ------------------
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
ax_2 = fig.add_subplot(gs[0,1])
ax_2.set_position(ax_2.get_position().translated(0, first_row_adjustment))
ax_2.errorbar(conditions, final_adpt_acc_mean, yerr=final_adpt_acc_std, capsize=3, fmt="r--o", ecolor = "black",markersize=4,color='tab:orange',alpha=0.5)
ax_2.spines['right'].set_visible(False)
ax_2.spines['top'].set_visible(False)
#ax_2.set_ylabel('Residual error [deg]')
ax_2.set_xlabel('Learning contribution \n (CB vs DA)')
ax_2.set_title('Rotation task: model predictions', fontsize=font_s)
ax_2.set_xticks([0,25,50,75,100])
ax_2.set_xticklabels(condition_labels)
#ax_2.xaxis.set_ticks_position('none') 
#ax_2.yaxis.set_ticks_position('none') 


## =============================== PLOT RESULTS FROM REVERSAl TASK ======================

main_file_dir = os.path.join(root_dir,'..','..','DartThrowing','results')

## ---------- Load human data on reversed adapt task, Gutierrez-Garralda et al., 2013 -------------
file_dir = os.path.join(main_file_dir,"human_BG_data.csv")
human_data = genfromtxt(file_dir, delimiter=',')
human_data = human_data[1:,:]

## Need to separate the data based on how they were extracted

# PD patients
PD_adpt = human_data[0,0]
PD_adapt_sde = np.abs(human_data[1,0] - PD_adpt)
PD_washout = human_data[0,1]
PD_washout_sde = np.abs(human_data[2,1] - PD_washout)

# HD patients
HD_adpt = human_data[3,0]
HD_adapt_sde = np.abs(human_data[4,0] - HD_adpt )
HD_washout = human_data[3,1]
HD_washout_sde = np.abs(human_data[5,1] - HD_washout )

# PD control
control_PD_adpt = human_data[6,0]
control_PD_adapt_sde = np.abs(human_data[7,0] - control_PD_adpt)
control_PD_washout = human_data[6,1]
control_PD_washout_sde = np.abs(human_data[8,1] - control_PD_washout)

# HD control
control_HD_adpt = human_data[9,0]
control_HD_adapt_sde = np.abs(human_data[10,0] - control_HD_adpt)
control_HD_washout = human_data[9,1]
control_HD_washout_sde = np.abs(human_data[11,1] - control_HD_washout)

##--To get data of BG patients vs control, sum results across PD and HD patients as well as two control groups
BG_patient_adpt = (HD_adpt + PD_adpt)/2
control_adpt = (control_HD_adpt + control_PD_adpt)/2

##--To compute the sde of a sum of variables need to convert sde back to variances and then summ the two variances and convert back to sde (assuming independence) 
HD_n_ppts = 24
PD_n_ppts = 17
control_n_ppts_1 = HD_n_ppts
control_n_ppts_2 = PD_n_ppts

#Patients adaptation sde
BG_patient_adapt_var = (np.sqrt(HD_n_ppts) * HD_adapt_sde)**2 + (np.sqrt(PD_n_ppts) * PD_adapt_sde)**2
BG_patient_adapt_sde = np.sqrt(BG_patient_adapt_var) / np.sqrt(HD_n_ppts + PD_n_ppts)

#Control adaptation sde
control_adapt_var = (np.sqrt(control_n_ppts_1) * control_HD_adapt_sde)**2 + (np.sqrt(control_n_ppts_2) * control_PD_adapt_sde)**2
control_adapt_sde = np.sqrt(control_adapt_var) / np.sqrt(control_n_ppts_1 + control_n_ppts_2)

##------- Load model data for both barchart (comparison with humans) and continous predictions across \beta
seeds = [8721, 5467, 1092, 9372,2801]

file_dir = os.path.join(main_file_dir,"reversal")

optBeta_seed_acc = [] # NOTE: to reproduce BG patient data used beta = 0.65, which I denote as optimal beta
RBL_seed_acc = []
Mixed_25_seed_acc = []
Mixed_50_seed_acc = []
Mixed_75_seed_acc = []
EBL_seed_acc = []
for s in seeds:

    # ----- Load model data to reproduce human data -------
    RBL_acc_dir = os.path.join(file_dir,str(s),"Reversed_Mixed_0.0accuracy.npy")
    ## NOTE: \beta=0.65 seems to best describe BG data
    optBeta_acc_dir = os.path.join(file_dir,str(s),"Reversed_Mixed_0.65accuracy.npy")

    RBL_acc = np.load(RBL_acc_dir)
    RBL_seed_acc.append(RBL_acc)
    optBeta_acc = np.load(optBeta_acc_dir)
    optBeta_seed_acc.append(optBeta_acc)
    ## -----------------------------------------

    # ------------ Load model data across different \beta
    Mixed25_acc_dir = os.path.join(file_dir,str(s),"Reversed_Mixed_0.25accuracy.npy")
    Mixed50_acc_dir = os.path.join(file_dir,str(s),"Reversed_Mixed_0.5accuracy.npy")
    Mixed75_acc_dir = os.path.join(file_dir,str(s),"Reversed_Mixed_0.75accuracy.npy")
    EBL_acc_dir = os.path.join(file_dir,str(s),"Reversed_Mixed_1.0accuracy.npy")

    Mixed_25_seed_acc.append(np.load(Mixed25_acc_dir))
    Mixed_50_seed_acc.append(np.load(Mixed50_acc_dir))
    Mixed_75_seed_acc.append(np.load(Mixed75_acc_dir))
    EBL_seed_acc.append(np.load(EBL_acc_dir))
    ## ---------------------------------------------

# Define useful variables
optBeta_seed_acc = np.array(optBeta_seed_acc) # optimal beta to reproduce BG data
RBL_seed_acc = np.array(RBL_seed_acc)
## --------------------------------------------

## -------------- Plot barchart comparison model and human data --------------- 
# Experimental variables
baseline_trials = 26
pertubed_trials = 26
washout_trials = 26
tot_trials = baseline_trials + pertubed_trials + washout_trials
final_prtb_trial = baseline_trials + pertubed_trials -2

# NOTE: I multiply model results by 100 since original paper results are reported in CM and not meters!
RBL_adpt =  100*(RBL_seed_acc[:,baseline_trials] - RBL_seed_acc[:,final_prtb_trial])
optBeta_adpt =  100*(optBeta_seed_acc[:,baseline_trials] - optBeta_seed_acc[:,final_prtb_trial])

# COmpute mean
RBL_adpt_mean = RBL_adpt.mean()
optBeta_adpt_mean = optBeta_adpt.mean()

# COmpute sde
RBL_adpt_sde = RBL_adpt.std() / np.sqrt(5)
optBeta_adpt_sde = optBeta_adpt.std() / np.sqrt(5)

model_adpt_means = [RBL_adpt_mean, optBeta_adpt_mean]
human_adpt_means = [control_adpt, BG_patient_adpt]

model_adpt_sde = [RBL_adpt_sde, optBeta_adpt_sde]
human_adpt_sde = [control_adapt_sde, BG_patient_adapt_sde]

#conditions = ['Control \n (DA driven)', 'BG patients \n (DA-deficiency)']
conditions = ['Control', 'BG patients']
labels = ['DA-driven \n (model)', 'human control', 'DA-deficiency \n (model)', 'BG patients']
colors = ['tab:blue','tab:red','tab:green']
x = np.array([0,1])
width=0.3
ax_10 = fig.add_subplot(gs[0,2])
ax_10.set_position(ax_10.get_position().translated(0, first_row_adjustment))
#for i in range(2):
ax_10.bar(x[0], model_adpt_means[0], width=width, align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k', color=colors[1]) #color='tab:gray',
ax_10.bar(x[0]+width, human_adpt_means[0], width=width, align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k', color='tab:gray')
ax_10.bar(x[1], model_adpt_means[1], width=width, align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k', color=colors[2], label=labels[2]) #color='tab:gray',
ax_10.bar(x[1]+width, human_adpt_means[1], width=width, align='center', alpha=0.5,ecolor='black', capsize=5, edgecolor='k', color='tab:purple', label=labels[3])
ax_10.errorbar(conditions[:2], model_adpt_means, yerr=model_adpt_sde, ls='none', color='black',  elinewidth=0.75, capsize=1.5) # ecolor='lightslategray',
ax_10.errorbar(x+ width, human_adpt_means, yerr=human_adpt_sde, ls='none', color='black',  elinewidth=0.75, capsize=1.5) # ecolor='lightslategray',
ax_10.axhline(0, color='black', linewidth=1)
ax_10.spines['bottom'].set_visible(False)
ax_10.set_xticks(x + width/2)
ax_10.set_xticklabels(conditions[:2])
ax_10.set_ylim([-15, 35])
#ax_10.set_yticks([0,5,10,15])
ax_10.spines['right'].set_visible(False)
ax_10.spines['top'].set_visible(False)
ax_10.set_ylabel('Adaptation [cm]')
ax_10.set_title('Reversal task: model vs data', fontsize=font_s)
ax_10.xaxis.set_ticks_position('none') 
#ax_10.yaxis.set_ticks_position('none') 
ax_10.legend(loc='upper left', bbox_to_anchor=(0.2, legend_height), frameon=False,fontsize=font_s, ncol=2)


## --------------------- Model predictions across \beta for reversal task -------------
# Model predicted adaptation at the final perturbation trial (i.e., amount of adaptation)
Mixed_25_seed_acc = np.array(Mixed_25_seed_acc)
Mixed_50_seed_acc = np.array(Mixed_50_seed_acc)
Mixed_75_seed_acc = np.array(Mixed_75_seed_acc)
EBL_seed_acc = np.array(EBL_seed_acc)

# NOTE: I again multiply model results by 100 to represent them in CM and not meters
RBL_error =  100 * RBL_seed_acc[:,final_prtb_trial]
Mixed25_error =  100 * Mixed_25_seed_acc[:,final_prtb_trial]
Mixed50_error =  100 * Mixed_50_seed_acc[:,final_prtb_trial]
Mixed75_error =  100 * Mixed_75_seed_acc[:,final_prtb_trial]
EBL_error =  100 * EBL_seed_acc[:,final_prtb_trial]

adpt_betas_mean = [RBL_error.mean(), Mixed25_error.mean(), Mixed50_error.mean(), Mixed75_error.mean(), EBL_error.mean()]
adpt_betas_sde = [RBL_error.std(), Mixed25_error.std(), Mixed50_error.std(), Mixed75_error.std(), EBL_error.std()] / np.sqrt(5)

conditions = [0,25,50,75,100]
condition_labels = ['0%', '25%','50%','75%', '100%']
ax_12 = fig.add_subplot(gs[0,3])
ax_12.set_position(ax_12.get_position().translated(0, first_row_adjustment))
ax_12.errorbar(conditions, adpt_betas_mean, yerr=adpt_betas_sde, capsize=3, fmt="r--o", ecolor = "black",markersize=4,color='tab:orange',alpha=0.5)
ax_12.spines['right'].set_visible(False)
ax_12.spines['top'].set_visible(False)
ax_12.set_ylabel('Residual error [cm]')
ax_12.set_xlabel('Learning contribution \n (CB vs DA)')
ax_12.set_title('Reversal task: model predictions', fontsize=font_s)
ax_12.set_xticks([0,25,50,75,100])
ax_12.set_xticklabels(condition_labels)
#ax_12.xaxis.set_ticks_position('none') 
#ax_12.yaxis.set_ticks_position('none') 


## ================= Plot dysmetria scores =======================
## Load data across different betas
data_dir = os.path.join(root_dir,'perturbations')
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

outcomes = [RBL_xy_outcome, Mix25_xy_outcome,Mix50_xy_outcome,Mix75_xy_outcome,EBL_xy_outcome]

sampled_trials = np.arange(0,10) * (-1)
# Add bottom right subplot - gs[bottom row, last column (the 'left' subplot)]
# We do not add the upper right subplot
dysmetria_mean_score = []
dysmetria_std_score = []
sde_norm = np.sqrt(len(sampled_trials) * 5 * 3) # 5: seeds, 3 targets
for a in outcomes:
    # NOTE: dysmetria seems to refer to over-shooting (hypermetria) and under-shooting (hypometria)
    # So compute score only relative to the displacement in x-coords
    x_displacement = np.sqrt((targets[...,0] - a[...,0])**2)
    # Compute mean and std across seeds and last n. trials (i.e., sampled_trials)
    mean_x_displ = x_displacement[:,sampled_trials].mean()
    std_x_displ = x_displacement[:,sampled_trials].std() / sde_norm
    #mean_y_displacement = ((targets[...,1] - a[...,1])**2).mean(axis=0).mean(axis=-1)
    dysmetria_mean_score.append(mean_x_displ)
    dysmetria_std_score.append(std_x_displ)

ax = fig.add_subplot(gs[1,0])
conditions = [0,25,50,75,100]
condition_labels = ['0%', '25%','50%','75%', '100%']
ax.errorbar(conditions, dysmetria_mean_score, yerr=dysmetria_std_score, capsize=3, fmt="r--o", ecolor = "black",markersize=4,color='tab:orange',alpha=0.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Dysmetria score')
ax.set_xlabel('Learning contribution \n ("dysfunctional-CB" vs DA)')
ax.set_title('Predicted dysmetria', fontsize=font_s)
ax.set_xticks([0,25,50,75,100])
ax.set_xticklabels(condition_labels)
#ax.xaxis.set_ticks_position('none') 
#ax.yaxis.set_ticks_position('none') 


## ------- Plot angle accuracy -------------
angle_acc = np.array([RBL_angle_acc[:,sampled_trials], Mix25_angle_acc[:,sampled_trials],Mix50_angle_acc[:,sampled_trials],Mix75_angle_acc[:,sampled_trials],EBL_angle_acc[:,sampled_trials]])
angle_acc = angle_acc.reshape(5,-1)
sde_norm = np.sqrt(len(sampled_trials)*5)

mean_angle_acc = angle_acc.mean(axis=-1)
std_angle_acc = angle_acc.std(axis=-1)/sde_norm

ax = fig.add_subplot(gs[1,1])
conditions = [0,1]
CB_driven_mean = mean_angle_acc[-1]
DA_driven_mean = mean_angle_acc[0]

CB_driven_std = std_angle_acc[-1]
DA_driven_std = std_angle_acc[0]

ax.errorbar(conditions[0], CB_driven_mean, yerr=CB_driven_std, capsize=3, fmt="r--o", ecolor = "black",markersize=4,color='tab:blue',alpha=0.5)
ax.errorbar(conditions[1], DA_driven_mean, yerr=DA_driven_std, capsize=3, fmt="r--o", ecolor = "black",markersize=4,color='tab:red',alpha=0.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Error [deg]')
#ax.set_xlabel('lesioned-CB contribution')
ax.set_xticks([0,1])
ax.set_xticklabels(['"dysfunctional-CB" \n driven', 'DA driven'])
ax.set_title('DA reduces CB-driven  impairments', fontsize=font_s)
#ax.xaxis.set_ticks_position('none') 
#ax.yaxis.set_ticks_position('none') 



## ================================= PLOT RESULTS FROM dy/da PERTURBATIONS ======================================


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
titles = ['Healthy', 'Hypermetria', 'Hypometria', 'Displacement']
first_row_adjustment = -0.02
for o in mean_xy_outcomeS:
    if i <2:
        ax = fig.add_subplot(gs[2,i])
        ax.set_position(ax.get_position().translated(0, first_row_adjustment))
        i+=1
    else:
        ax = fig.add_subplot(gs[3,e])
        e+=1
    sampled_trials = np.arange(0,10) * (-1)
    x_traj = o[sampled_trials,...,0]#, indx_target_plotted]
    y_traj = o[sampled_trials,...,1]#, indx_target_plotted]
    plot_target_indx = 2
    # Plot endpoints for each target
    ax.scatter(x_traj[:,plot_target_indx], y_traj[:,plot_target_indx], color='tab:blue', s=15, alpha=0.65, marker='x', label='Reaching outcome')
    # Plot targets
    ax.scatter(x_targ[plot_target_indx], y_targ[plot_target_indx], color='tab:red', s=25, alpha=0.65, marker='*', label='Target')
    #ax.set_ylim([0.5,0.7])
    ax.set_xlim([-0.1,0.09])
    ax.set_ylim([0.56,0.64])
    #ax.set_xlim([-0.1,0.09])
    if e>0:
        ax.set_xlabel("x-axis")
    if i == 1 or e==1:
        ax.set_ylabel("y-axis")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(titles[i+e-1],fontsize=font_s)#,fontweight='extra bold')
    if i == 0 or e==0:
        ax.legend(loc='lower left', bbox_to_anchor=(-1.65, 0.6), frameon=False,fontsize=font_s, ncol=1)

## ==========================================================


## ================ Plot reaching task with binary feedback =====

data_dir = os.path.join(root_dir,'..','..','binary_reaching','results')
DA_reduction = [1,0.1, 0.01, 0.001]

mean_final_acc = []
mean_initial_std = []
mean_final_std = []
mean_initial_mu = []
mean_final_mu = []

std_final_acc = []
std_initial_std = []
std_final_std = []
std_difference = []

# loop around condition of DA reduction
for i in range(len(DA_reduction)):
    label = "DA_decrease_"+str(i)+'.json'
    result_dir = os.path.join(data_dir,label) # For the mixed model

    # Open and read the JSON file
    with open(result_dir, 'r') as file:
            results = json.load(file)

    mean_accuracy = (np.array(results['accuracy'])*100).mean(axis=0) # convert to %
    mean_std = np.array(results['std_a']).mean(axis=0)

    # Compute difference between intial action and std and final action std for each seed
    std_a_difference = np.abs(np.array(results['std_a'])[:,0] - np.array(results['std_a'])[:,-1])
    std_difference.append(std_a_difference)

    std_accuracy = (np.array(results['accuracy'])*100).std(axis=0)
    std_std = np.array(results['std_a']).std(axis=0)

    mean_mu = np.array(results['mu_a']).mean(axis=0)

    # Store value for each condition
    mean_final_acc.append(mean_accuracy[-1])
    mean_initial_std.append(mean_std[0])
    mean_final_std.append(mean_std[-1])
    mean_initial_mu.append(mean_mu[0])
    mean_final_mu.append(mean_mu[-1])

    std_final_acc.append(std_accuracy[-1])
    std_initial_std.append(std_std[0])
    std_final_std.append(std_std[-1])

# ----- Plot accuracy across DA reduction ----
conditions = [1,2,3,4]
condition_labels = ['x1','x10','x100','x1000']
ax_2 = fig.add_subplot(gs[1,2])
#ax_2.set_position(ax_2.get_position().translated(0, first_row_adjustment))
ax_2.errorbar(conditions, mean_final_acc, yerr=std_final_acc, capsize=3, fmt="r--o", ecolor = "black",markersize=4,color='tab:orange',alpha=0.5)
ax_2.spines['right'].set_visible(False)
ax_2.spines['top'].set_visible(False)
#ax_2.set_ylabel('Residual error [deg]')
ax_2.set_xlabel('DA reduction')
ax_2.set_ylabel('Accuracy')
ax_2.set_title('DA deficits with discrete-feedback', fontsize=font_s)
ax_2.set_yticks([0,25,50,75,100])
ax_2.set_xticks([1,2,3,4])
ax_2.set_xticklabels(condition_labels)
ax_2.set_yticklabels(['0%','25%','50%','75%','100%'])
#ax_2.xaxis.set_ticks_position('none') 

# ----- Plot std reduction from start to end of learning across DA reduction ----

mean_std_a_diff = np.array(std_difference).mean(axis=-1)
std_std_a_diff = np.array(std_difference).std(axis=-1)
conditions = [1,2,3,4]
condition_labels = ['x1','x10','x100','x1000']
ax_2 = fig.add_subplot(gs[1,3])
#ax_2.set_position(ax_2.get_position().translated(0, first_row_adjustment))
ax_2.errorbar(conditions, mean_std_a_diff, yerr=std_std_a_diff, capsize=3, fmt="r--o", ecolor = "black",markersize=4,color='tab:orange',alpha=0.5)
ax_2.spines['right'].set_visible(False)
ax_2.spines['top'].set_visible(False)
#ax_2.set_ylabel('Residual error [deg]')
ax_2.set_xlabel('DA reduction')
ax_2.set_ylabel('Uncertainty reduction')
ax_2.set_title('DA deficits and action uncertainty', fontsize=font_s)
#ax_2.set_yticks([0,25,50,75,100])
ax_2.set_xticks([1,2,3,4])
ax_2.set_xticklabels(condition_labels)
#ax_2.set_yticklabels(['0%','25%','50%','75%','100%'])

## ----- Plot action distribtuion shift -----
i=2
e=2
first_row_adjustment = -0.02
colors = ['tab:red','tab:red','tab:green','tab:green']
for t in range(4):
    if t <2:
        ax = fig.add_subplot(gs[2,i])
        ax.set_position(ax.get_position().translated(0, first_row_adjustment))
        i+=1
    else:
        ax = fig.add_subplot(gs[3,e])
        e+=1

    mean = 0

    # Select appropriate std to plot
    if t ==0:
        #mean = mean_initial_mu[0] 
        std_dev = mean_initial_std[0] 
    elif t ==1:
        #mean = mean_final_mu[0] 
        std_dev = mean_final_std[0] 
    elif t==2:
        #mean = mean_initial_mu[-1] 
        std_dev = mean_initial_std[-1] 
    else:
        #mean = mean_final_mu[-1] 
        std_dev = mean_final_std[-1] 
    # Create x values: typically ±3 standard deviations around the mean
    x = np.linspace(mean - 2 * std_dev, mean + 2 * std_dev, 1000)
    # Calculate the Gaussian PDF for each x value
    pdf = norm.pdf(x, mean, std_dev)

    # Plot
    ax.plot(x, pdf,  color=colors[t], alpha=0.75, linewidth=0.75)

    if t>1:
        ax.set_xlabel("Reaching angle")
    if t ==0:
        ax.set_title("Early learning",fontsize=font_s)
    if t ==1:
        ax.set_title("Late learning",fontsize=font_s)
    if t == 0: 
        ax.set_ylabel("Healthy", fontsize=9)
    elif t ==2:
        ax.set_ylabel("DA deficency", fontsize=9)
    #else:
    #    ax.spines['left'].set_visible(False)
    ax.set_xticks([-0.35,0,0.35])
    ax.set_xticklabels(['-0.3','$\mu_a$','0.3'])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([-0.35,0.35])
    #ax.set_title(titles[i+e-1],fontsize=font_s)#,fontweight='extra bold')
    #if i == 0 or e==0:
    #    ax.legend(loc='lower left', bbox_to_anchor=(-1.65, 0.6), frameon=False,fontsize=font_s, ncol=1)


#plt.tight_layout()

if save_file:
    plt.savefig('/Users/px19783/Desktop/paper_plot_2.png', format='png', dpi=1400)
else:
    plt.show()
