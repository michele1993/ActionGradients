import os
import argparse
from Linear_motor_model  import Mot_model
from Agent import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from CombinedAG import CombActionGradient

""" Load a pre-trained model and test reversed perturbation matching Gutierrez-Garralda et al., 2013 paper for BG patients (i.e., \beta = 1)
"""

parser = argparse.ArgumentParser()
parser.add_argument('--beta', '-b',type=float, nargs='?')

## Argparse variables:
args = parser.parse_args()
beta = args.beta

seeds = [8721, 5467, 1092, 9372,2801]
save_file = True

# Experimental set-up based on 
baseline_trials = 26
pertubed_trials = 26
washout_trials = 26
tot_trials = baseline_trials + pertubed_trials + washout_trials

perturbation_increase = 0.0176 # equivalent to 1 degree
max_perturbation = perturbation_increase 
target = 0.0176 # centered at sholuder location
y_star = torch.tensor([target],dtype=torch.float32)
## A horizontal reversing of the perceived position of the target by 11.31 degrees to the right with respect to the real target
MR_line = torch.tensor([11.31 * (2 * np.pi / 360)],dtype=torch.float32) # if the target is at 0 degrees then mirror-line should be placed at 11.31 to the right of the target
delta_MR = y_star - MR_line
reversed_y_star = MR_line - delta_MR

# Set noise variables
sensory_noise = 0.01
fixd_a_noise = 0.02#0.02 # set to experimental data value

# Set update variables
a_ln_rate = 0.02
c_ln_rate = 0.1
model_ln_rate = 0.5
rbl_weight = [1, 1]
ebl_weight = [1, 1]



# Load models
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'results') # For the mixed model
model_dir = os.path.join(file_dir,'model','Mixed_model.pt') # For the mixed model
#model_dir = os.path.join(file_dir,'model','RBL_model.pt') # For the mixed model
models = torch.load(model_dir)


for s in seeds:
    torch.manual_seed(s)
    # Store results in new directory within results
    acc_dir = os.path.join(file_dir,'reversal',str(s))
    os.makedirs(acc_dir, exist_ok=True)
    ## Reinitialise all the models for each run with a different beta
    actor = Actor(ln_rate = a_ln_rate, trainable = True, opt_type='SGD')
    actor.load_state_dict(models['Actor'])
    estimated_model = Mot_model(ln_rate=model_ln_rate,lamb=None,Fixed=False)
    estimated_model.load_state_dict(models['Est_model'])
    mean_rwd = models['Mean_rwd']

    # Initialise additional components
    model = Mot_model()
    CAG = CombActionGradient(actor=actor, beta=beta, rbl_weight=rbl_weight, ebl_weight=ebl_weight)

    tot_accuracy = []
    tot_actions = []
    tot_outcomes = []
    
    current_y_star = y_star
    for ep in range(1,tot_trials):

        ## ====== Apply perturbation to the target =======
        # need to be applied to the target, before reaching!!
        if ep > baseline_trials and ep < (baseline_trials + pertubed_trials): 
            current_y_star = reversed_y_star
        else:
            current_y_star = y_star
        ## ==============================================

        # Sample action from Gaussian policy
        action, mu_a, std_a = actor.computeAction(current_y_star, fixd_a_noise)

        # Perform action in the env
        true_y = model.step(action.detach())

        #MR_line = y_star + (y_star - true_y)/2
        
        ## ====== Apply perturbation to the outcome =======
        if ep > baseline_trials and ep < (baseline_trials + pertubed_trials): 
            delta_MR = true_y - MR_line
            obs_y = MR_line - delta_MR
        else:
            obs_y = true_y
        ## ==============================================

        # Add noise to sensory obs
        y = obs_y + torch.randn_like(true_y) * sensory_noise 

        # Compute differentiable rwd signal
        y.requires_grad_(True)
        rwd = (y - current_y_star)**2 # it is actually a punishment
        true_rwd = (true_y - y_star)**2 # it is actually a punishment
        
        ## ====== Use running average to compute RPE =======
        delta_rwd = rwd - mean_rwd
        mean_rwd += c_ln_rate * delta_rwd.detach()
        ## ==============================================

        # Update the model
        est_y = estimated_model.step(action.detach())
        model_loss = estimated_model.update(y, est_y)

        # Update actor based on combined action gradient
        #if ep > pre_train:
        est_y = estimated_model.step(action)  # re-estimate values since model has been updated
        CAG.update(y, est_y, action, mu_a, std_a, delta_rwd, delta_rwd) # pass the RPE as error and rwd

        # Store variables 
        accuracy = np.sqrt(true_rwd.detach().numpy())
        tot_accuracy.append(accuracy)
        tot_actions.append(action.detach().numpy())
        tot_outcomes.append(true_y.detach().numpy() - target) # make it so that 0 radiants refer to the target

    #print(seed_accuracy[baseline_trials])
    #print(seed_accuracy[baseline_trials+1])

    ### ===== Plot accuracy =======
    #if s == seeds[0]:
    #    t = np.arange(0, len(tot_accuracy))
    #    plt.scatter(t,tot_accuracy)
    #    plt.show()
    #    exit()
    ### ==========================

    label = "Reversed_Mixed_"+str(beta)
    outcome_tot_dir = os.path.join(acc_dir,label+'accuracy') # For the mixed model

    # Save all outcomes so that can then plot whatever you want
    if save_file: 
        # Only store final variability
        np.save(outcome_tot_dir, tot_accuracy)
