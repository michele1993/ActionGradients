import os
from Linear_motor_model  import Mot_model
from Agent import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from CombinedAG import CombActionGradient

" Generate policy with minimal noise across betas to test generalisation performance"

seeds = [8721, 5467, 1092, 9372,2801]
trials = 4000
t_print = 100
save_file = False

## ===== All the variables below set equal to the motor variability experiments as based on the same Izawa's set-up ====
# Set noise variables
sensory_noise = 0.01
fixd_a_noise = 0.018 # set to experimental data value

rwd_area = 0.01 

# Set update variables
a_ln_rate = 0.05
c_ln_rate = 0.05
model_ln_rate = 0.01
betas = np.arange(0,11,1) /10.0
rbl_weight = [0.002, 5] # [0.034, 100]
ebl_weight = [0.2, 5] # [1.5, 100]
## =================================


## Peturbation:
#targets = [-30, -20, -10, 0, 10, 20, 30] # based on Izawa
targets = [-1,1]
y_star = torch.tensor(targets,dtype=torch.float32).unsqueeze(-1) * 0.0176

model = Mot_model()

for s in seeds:
    torch.manual_seed(s)

    for b in betas:
        # Initialise differento components
        actor = Actor(action_s=1, ln_rate = a_ln_rate, trainable = True) # 1D environment
        estimated_model = Mot_model(ln_rate=model_ln_rate,lamb=None, Fixed = False)
        CAG = CombActionGradient(actor, b, rbl_weight, ebl_weight)

        tot_accuracy = []
        mean_rwd = 0
        trial_acc = []

        for ep in range(1,trials+1):
            # Sample action from Gaussian policy
            action, mu_a, std_a = actor.computeAction(y_star, fixd_a_noise)

            # Perform action in the env
            true_y = model.step(action.detach())
            
            # Add noise to sensory obs
            y = true_y + torch.randn_like(true_y) * sensory_noise 

            # Compute differentiable rwd signal
            y.requires_grad_(True)
            error = (y - y_star)**2 # it is actually a punishment
            trial_acc.append(torch.sqrt(error.detach()).mean().item())

            ## ====== Give a rwd if reach is within target area
            #rwd = (torch.sqrt(error) <= rwd_area).int()
            #rwd[rwd==0] = -1
            ## ==============================================

            ## ====== Use running average to compute RPE =======
            delta_rwd = error - mean_rwd
            mean_rwd += c_ln_rate * delta_rwd.detach()
            #NOTE: we provide a dense rwd signal since for a policy to learn task completely from scratch based on success/fail is hard
            delta_rwd[delta_rwd>0] = 1 
            delta_rwd[delta_rwd<0] = -1 
            ## ==============================================

            # Update the model
            est_y = estimated_model.step(action.detach())
            model_loss = estimated_model.update(y, est_y)

            # Update actor based on combined action gradient
            est_y = estimated_model.step(action)  # re-estimate values since model has been updated
            CAG.update(y, est_y, action, mu_a, std_a, error, delta_rwd)

            # Store variables after pre-train (including final trials without a perturbation)
            if ep % t_print ==0:
                accuracy = sum(trial_acc) / len(trial_acc)
                #print("ep: ",ep)
                #print("accuracy: ",accuracy)
                #print("std_a: ", std_a,"\n")
                tot_accuracy.append(accuracy)
                trial_acc = []

        #print("agent mu weight: ", actor.l1.weight.item())
        #print("agent mu bias: ", actor.l1.bias.item())
        #print("agent std weight: ", actor.l2.weight.item())
        #print("agent std bias: ", actor.l2.bias.item())

        print("Beta: ",b, "accuracy: ", accuracy)

        ## ===== Save results =========
        # Create directory to store results
        file_dir = os.path.dirname(os.path.abspath(__file__))
        file_dir = os.path.join(file_dir,'results',str(s))
        # Create directory if it did't exist before
        os.makedirs(file_dir, exist_ok=True)

        # Store model
        data = 'Mixed_'+str(b)+'model.pt'
        model_dir = os.path.join(file_dir,data)

        if save_file:
            torch.save({
                "Training_acc": tot_accuracy,
                "Targets": targets,
                'Actor': actor.state_dict(),
                'Net_optim': actor.optimiser.state_dict(),
                'Mean_rwd': mean_rwd,
                'Est_model': estimated_model.state_dict(),
                'Model_optim': estimated_model.optimiser.state_dict(),
            }, model_dir)
