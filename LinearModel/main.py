import os
from Motor_model  import Mot_model
from Agent import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from CombinedAG import CombActionGradient

torch.manual_seed(0)

trials = 2500
t_print = 100
save_file = True
## Set trials to match Izawa and Shadmer, 2011 experimental set-up, where they add 1 degree pertubation every 40 trials up to 8 degreese

# Set noise variables
sensory_noise = 0.01
fixd_a_noise = 0.02 # set to experimental data value

# Set update variables
a_ln_rate = 0.01
c_ln_rate = 0.1
model_ln_rate = 0.01
beta_mu = 0.5
beta_std = beta_mu
rbl_std_weight =  1.5
ebl_std_weight = 0.1

## Peturbation:

target = 0.1056 # target angle : 6 degrees - Izawa and Shadmer, 2011
y_star = torch.tensor([target],dtype=torch.float32)

model = Mot_model()

actor = Actor(output_s=2, ln_rate = a_ln_rate, trainable = True)
estimated_model = Mot_model(ln_rate=model_ln_rate,lamb=None, Fixed = False)

CAG = CombActionGradient(actor, beta_mu, beta_std, rbl_std_weight, ebl_std_weight)

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
    rwd = (y - y_star)**2 # it is actually a punishment
    trial_acc.append(torch.sqrt(rwd.detach()).item())
    
    ## ====== Use running average to compute RPE =======
    delta_rwd = rwd - mean_rwd
    mean_rwd += c_ln_rate * delta_rwd.detach()
    ## ==============================================

    # For rwd-base learning give rwd of 1 if reach better than previous else -1
    if beta_mu == 0:
       delta_rwd /= torch.abs(delta_rwd.detach()) 


    # Update the model
    est_y = estimated_model.step(action.detach())
    model_loss = estimated_model.update(y, est_y)

    # Update actor based on combined action gradient
    est_y = estimated_model.step(action)  # re-estimate values since model has been updated
    CAG.update(y, est_y, action, mu_a, std_a, delta_rwd)

    # Store variables after pre-train (including final trials without a perturbation)
    if ep % t_print ==0:
        accuracy = sum(trial_acc) / len(trial_acc)
        print("ep: ",ep)
        print("accuracy: ",accuracy)
        print("std_a: ", std_a,"\n")
        tot_accuracy.append(accuracy)
        trial_acc = []

## ===== Save results =========
# Create directory to store results
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'results/model')
# Create directory if it did't exist before
os.makedirs(file_dir, exist_ok=True)

# Store model
if beta_mu ==0:
    data = 'RBL_model.pt'
elif beta_mu ==1:    
    data = 'EBL_model.pt'
else:
    data = 'Mixed_model.pt'
model_dir = os.path.join(file_dir,data)

if save_file:
    torch.save({
        'Actor': actor.state_dict(),
        'Net_optim': actor.optimiser.state_dict(),
        'Mean_rwd': mean_rwd,
        'Est_model': estimated_model.state_dict(),
        'Model_optim': estimated_model.optimiser.state_dict(),
    }, model_dir)
