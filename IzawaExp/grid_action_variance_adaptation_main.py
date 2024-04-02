import os
from Linear_motor_model  import Mot_model
from Agent import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from CombinedAG import CombActionGradient

"""Fix the RPEs to either only positive or only negative to study how the action variance is adpated in response to neg and pos RPEs"""

seeds = [8721, 5467, 1092, 9372,2801]
save_file = False

# Experimental set-up based on Izawa and Shadmer, 2011
tot_trials = 101
t_print = 10 # how often compute mean of std_a to be stored
target = 0.1056 # target angle : 6 degrees 
y_star = torch.tensor([target],dtype=torch.float32)
neg_RPE = True # use to fix RPE to either all positive or all negative

# Set noise variables
sensory_noise = 0.01
fixd_a_noise = 0.02 #0.02 # set to experimental data value

# Set update variables
a_ln_rate = 0.001
c_ln_rate = 0.01
model_ln_rate = 0.01
betas = [0] #np.arange(0,11,1) /10.0
rbl_weight = [0.01,0.01] #[0.01, 0.01]
ebl_weight = [5, 100]


# Load models
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'results') # For the mixed model
model_dir = os.path.join(file_dir,'model','Mixed_model.pt') # For the mixed model
models = torch.load(model_dir)

for beta in betas:
    seed_std_a = []
    for s in seeds:

        torch.manual_seed(s)

        ## Reinitialise all the models for each run with a different beta
        actor = Actor(ln_rate = a_ln_rate, trainable = True)#, opt_type='SGD')
        actor.load_state_dict(models['Actor'])
        estimated_model = Mot_model(ln_rate=model_ln_rate,lamb=None,Fixed=False)
        estimated_model.load_state_dict(models['Est_model'])
        mean_rwd = models['Mean_rwd']
        #mean_rwd = 0 

        # Initialise additional components
        model = Mot_model()
        CAG = CombActionGradient(actor=actor, beta=beta, rbl_weight=rbl_weight, ebl_weight=ebl_weight)

        tot_std_a = []
        trial_std_a = []

        for ep in range(0,tot_trials):

            # Sample action from Gaussian policy
            action, mu_a, std_a = actor.computeAction(y_star, fixd_a_noise)
            trial_std_a.append(std_a.detach().squeeze().numpy())

            # Perform action in the env
            true_y = model.step(action.detach())
            
            # Add noise to sensory obs
            y = true_y + torch.randn_like(true_y) * sensory_noise 


            # Compute differentiable rwd signal
            y.requires_grad_(True)
            rwd = (y - y_star)**2 # it is actually a punishment
            
            ## ====== Use running average to compute RPE =======
            delta_rwd = rwd - mean_rwd
            mean_rwd += c_ln_rate * delta_rwd.detach()
            ## ==============================================

            # For rwd-base learning give rwd of 1 if reach better than previous else -1
            if beta == 0:
               delta_rwd /= (torch.abs(delta_rwd.detach()) + 1e-12)
               if neg_RPE:
                    delta_rwd = - torch.abs(delta_rwd)
               else: 
                    delta_rwd = torch.abs(delta_rwd)

            # Update the model
            est_y = estimated_model.step(action.detach())
            model_loss = estimated_model.update(y, est_y)

            # Update actor based on combined action gradient
            #if ep > pre_train:
            est_y = estimated_model.step(action)  # re-estimate values since model has been updated
            CAG.update(y, est_y, action, mu_a, std_a, delta_rwd)

            # Store mean action std
            if ep % t_print == 0:
                mean_sdt_a = sum(trial_std_a) /len(trial_std_a)
                tot_std_a.append(mean_sdt_a)
                trial_std_a = []

        seed_std_a.append(tot_std_a)

    seed_std_a = np.array(seed_std_a)
    #mean_std_a = seed_std_a.mean(axis=0)
    #std_std_a = seed_std_a.std(axis=0)

    # Store results in new directory within results
    file_dir = os.path.join(file_dir,'beta_grid')
    os.makedirs(file_dir, exist_ok=True)
    label = "Mixed_"+str(beta)
    if neg_RPE:
        std_var_dir = os.path.join(file_dir,label+'_std_a_adaptation_NegRPE') # For the mixed model
    else:
        std_var_dir = os.path.join(file_dir,label+'_std_a_adaptation_PosRPE') # For the mixed model

    # Save all outcomes so that can then plot whatever you want
    if save_file: 
        # Only store final variability
        np.save(std_var_dir, seed_std_a)
