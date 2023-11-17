import os
from Linear_motor_model  import Mot_model
from Agent import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from CombinedAG import CombActionGradient

""" Load a pre-trained model and test under perturbation matching Izawa and Shadmer, 2011 experimental set-up, where they add 1 degree pertubation every 40 trials up to 8 degreese , do these across different values for Beta as well as across 5 random seeds"""

seeds = [8721, 5467, 1092, 9372,2801]
save_file = False

# Experimental set-up based on Izawa and Shadmer, 2011
trials_x_perturbation = 40
baseline = trials_x_perturbation # Store the same amount of trials but without perturbation
n_pertubations = 8
perturb_trials = trials_x_perturbation * n_pertubations # 8 * 40 , n. perturb x trials_per_perturb
fixed_trials = 140 # final trials at 8 degree rotation pertub, based on which action variance is assessed (Note 140 because last pertub had 40 extra)
tot_trials = baseline + perturb_trials + fixed_trials
perturbation_increase = 0.0176 # equivalent to 1 degree
max_perturbation = perturbation_increase * n_pertubations
target = 0.1056 # target angle : 6 degrees 
y_star = torch.tensor([target],dtype=torch.float32)

# Set noise variables
sensory_noise = 0.01
fixd_a_noise = 0.019#0.02 # set to experimental data value

# Set update variables
a_ln_rate = 0.05
c_ln_rate = 0.05
model_ln_rate = 0.01
betas = np.arange(0,11,1) /10.0
rbl_weight = [0.01, 0.01]
ebl_weight = [5, 100]
#rbl_weight = [1, 1]
#ebl_weight = [1, 1]



# Load models
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'results') # For the mixed model
model_dir = os.path.join(file_dir,'model','Mixed_model.pt') # For the mixed model
models = torch.load(model_dir)



for s in seeds:
    torch.manual_seed(s)
    # Store results in new directory within results
    acc_dir = os.path.join(file_dir,'beta_grid',str(s))
    os.makedirs(acc_dir, exist_ok=True)
    for beta in betas:
        ## Reinitialise all the models for each run with a different beta
        actor = Actor(output_s=2, ln_rate = a_ln_rate, trainable = True, opt_type='SGD')
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

        # start with a zero perturbation
        current_perturbation = 0

        for ep in range(0,tot_trials):

            # Sample action from Gaussian policy
            action, mu_a, std_a = actor.computeAction(y_star, fixd_a_noise)

            # Perform action in the env
            true_y = model.step(action.detach())
            
            ## ====== Increase perturbation =======
            # Follow procedure from Izawa and Shadmer, 2011
            # after pre_train add a 1-degree perturbation every 40 trials upto 8 degree perturbation
            if ep >= baseline and ep % trials_x_perturbation == 1 and np.abs(current_perturbation) < max_perturbation:
               current_perturbation -= perturbation_increase
               #print("\n Ep: ", ep, "Perturbation: ", current_perturbation/0.0176, "\n")
            ## ==============================================

            # Add noise to sensory obs
            y = true_y + torch.randn_like(true_y) * sensory_noise 

            # Add current perturbation to observation
            y+= current_perturbation

            # Compute differentiable rwd signal
            y.requires_grad_(True)
            rwd = (y - y_star)**2 # it is actually a punishment
            
            ## ====== Use running average to compute RPE =======
            delta_rwd = rwd - mean_rwd
            mean_rwd += c_ln_rate * delta_rwd.detach()
            ## ==============================================

            # For rwd-base learning give rwd of 1 if reach better than previous else -1
            if beta == 0:
               delta_rwd /= torch.abs(delta_rwd.detach()) 

            # Update the model
            est_y = estimated_model.step(action.detach())
            model_loss = estimated_model.update(y, est_y)

            # Update actor based on combined action gradient
            #if ep > pre_train:
            est_y = estimated_model.step(action)  # re-estimate values since model has been updated
            CAG.update(y, est_y, action, mu_a, std_a, delta_rwd)

            # Store variables 
            accuracy = np.sqrt(rwd.detach().numpy())
            tot_accuracy.append(accuracy)
            tot_actions.append(action.detach().numpy())
            tot_outcomes.append(true_y.detach().numpy() - target) # make it so that 0 radiants refer to the target

        outcome_variability = np.std(tot_outcomes[-fixed_trials:])
        print("Beta: ", beta)
        print("Tot variability: ",outcome_variability,"\n") # compute variability across final fixed trials like in paper

        label = "Mixed_"+str(beta)
        outcome_var_dir = os.path.join(acc_dir,label+'_outcome_variability') # For the mixed model
        outcome_tot_dir = os.path.join(acc_dir,label+'_trajectories') # For the mixed model

        # Save all outcomes so that can then plot whatever you want
        if save_file: 
            # Only store final variability
            np.save(outcome_var_dir, outcome_variability)
            np.save(outcome_tot_dir, tot_outcomes)
