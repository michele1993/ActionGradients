import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import set_matplotlib_formats
from numpy import genfromtxt


data_dir = os.path.dirname(os.path.abspath(__file__))

## ---------- Load human data on reversed adapt task, Gutierrez-Garralda et al., 2013 -------------
file_dir = os.path.join(data_dir,"human_BG_data.csv")
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

## To get data of BG patients vs control, sum results across PD and HD patients as well as two control groups
BG_patient_adpt = (HD_adpt + PD_adpt)/2
BG_patient_washout = (HD_washout + PD_washout)/2
control_adpt = (control_HD_adpt + control_PD_adpt)/2
control_washout = (control_HD_washout + control_PD_washout)/2

## ------ To compute the sde of a sum of variables need to convert sde back to variances and then summ the two variances and convert back to sde (assuming independence) --------
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

#Patients washout sde
BG_patient_washout_var = (np.sqrt(HD_n_ppts) * HD_washout_sde)**2 + (np.sqrt(PD_n_ppts) * PD_washout_sde)**2
BG_patient_washout_sde = np.sqrt(BG_patient_washout_var) / np.sqrt(HD_n_ppts + PD_n_ppts)

#Control washout sde
control_washout_var = (np.sqrt(control_n_ppts_1) * control_HD_washout_sde)**2 + (np.sqrt(control_n_ppts_2) * control_PD_washout_sde)**2
control_adapt_sde = np.sqrt(control_washout_var) / np.sqrt(control_n_ppts_1 + control_n_ppts_2)
## ------------------------------------------------



## ----------- Load model data --------------------
seeds = [8721, 5467, 1092, 9372,2801]

file_dir = os.path.join(data_dir,"reversal")

RBL_seed_acc = []
optBeta_seed_acc = []
Mixed_25_seed_acc = []
Mixed_50_seed_acc = []
Mixed_75_seed_acc = []
EBL_seed_acc = []
for s in seeds:

    # ----- Load data to reproduce human data -------
    RBL_acc_dir = os.path.join(file_dir,str(s),"Reversed_Mixed_0.0accuracy.npy")
    ## NOTE: \beta=0.65 seems to best describe BG data
    optBeta_acc_dir = os.path.join(file_dir,str(s),"Reversed_Mixed_0.65accuracy.npy")

    RBL_acc = np.load(RBL_acc_dir)
    RBL_seed_acc.append(RBL_acc)
    optBeta_acc = np.load(optBeta_acc_dir)
    optBeta_seed_acc.append(optBeta_acc)
    ## -----------------------------------------

    # ------------ Load model prediction data
    Mixed25_acc_dir = os.path.join(file_dir,str(s),"Reversed_Mixed_0.25accuracy.npy")
    Mixed50_acc_dir = os.path.join(file_dir,str(s),"Reversed_Mixed_0.5accuracy.npy")
    Mixed75_acc_dir = os.path.join(file_dir,str(s),"Reversed_Mixed_0.75accuracy.npy")
    EBL_acc_dir = os.path.join(file_dir,str(s),"Reversed_Mixed_1.0accuracy.npy")

    Mixed_25_seed_acc.append(np.load(Mixed25_acc_dir))
    Mixed_50_seed_acc.append(np.load(Mixed50_acc_dir))
    Mixed_75_seed_acc.append(np.load(Mixed75_acc_dir))
    EBL_seed_acc.append(np.load(EBL_acc_dir))

optBeta_seed_acc = np.array(optBeta_seed_acc)
RBL_seed_acc = np.array(RBL_seed_acc)

Mixed_25_seed_acc = np.array(Mixed_25_seed_acc)
Mixed_50_seed_acc = np.array(Mixed_50_seed_acc)
Mixed_75_seed_acc = np.array(Mixed_75_seed_acc)
EBL_seed_acc = np.array(EBL_seed_acc)
## --------------------------------------------

## -------- Plot data comparison ----------
# Experimental variables
baseline_trials = 26
pertubed_trials = 26
washout_trials = 26
tot_trials = baseline_trials + pertubed_trials + washout_trials
final_prtb_trial = baseline_trials + pertubed_trials -2

RBL_adpt =  100*(RBL_seed_acc[:,baseline_trials] - RBL_seed_acc[:,final_prtb_trial])
optBeta_adpt =  100*(optBeta_seed_acc[:,baseline_trials] - optBeta_seed_acc[:,final_prtb_trial])
#print(RBL_seed_acc[0, baseline_trials:final_prtb_trial+1],'\n change \n', EBL_seed_acc[0,baseline_trials:final_prtb_trial+1])
#exit()
RBL_washout = 100*RBL_seed_acc[:,final_prtb_trial+1]
optBeta_washout = 100*optBeta_seed_acc[:,final_prtb_trial+1]
#print(EBL_seed_acc[0, final_prtb_trial:], RBL_seed_acc[0,final_prtb_trial:])
#exit()

# COmpute mean
RBL_adpt_mean = RBL_adpt.mean()
optBeta_adpt_mean = optBeta_adpt.mean()

RBL_washout_mean = RBL_washout.mean()
optBeta_washout_mean = optBeta_washout.mean()

# COmpute sde
RBL_adpt_sde = RBL_adpt.std() / np.sqrt(5)
optBeta_adpt_sde = optBeta_adpt.std() / np.sqrt(5)

RBL_washout_sde = RBL_washout.std() / np.sqrt(5)
optBeta_washout_sde = optBeta_washout.std() / np.sqrt(5)

print(RBL_adpt_mean, control_adpt)
print(optBeta_adpt_mean, BG_patient_adpt)
#print(RBL_washout_mean, control_washout)
#print(EBL_washout_mean, BG_patient_washout)

print('\n', BG_patient_adapt_sde)
print('\n', control_adapt_sde)


## --------- Plot model prediction ----
RBL_error =  100 * RBL_seed_acc[:,final_prtb_trial]
Mixed25_error =  100 * Mixed_25_seed_acc[:,final_prtb_trial]
Mixed50_error =  100 * Mixed_50_seed_acc[:,final_prtb_trial]
Mixed75_error =  100 * Mixed_75_seed_acc[:,final_prtb_trial]
EBL_error =  100 * EBL_seed_acc[:,final_prtb_trial]

print(RBL_error.mean(), Mixed25_error.mean(), Mixed50_error.mean(), Mixed75_error.mean(), EBL_error.mean())


#plt.scatter(PD_adpt,PD_washout)
#plt.scatter(HD_adpt,HD_washout)
#plt.scatter(control_PD_adpt,control_PD_washout)
#plt.scatter(control_HD_adpt,control_HD_washout)
#plt.show()
