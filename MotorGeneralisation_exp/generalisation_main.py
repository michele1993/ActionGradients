import os
from Linear_motor_model  import Mot_model
from Agent import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from CombinedAG import CombActionGradient

torch.manual_seed(0)

trials = 20000
t_print = 1000
save_file = True

# Set noise variables
sensory_noise = 0.01
fixd_a_noise = 0.02 # set to experimental data value

# Set update variables
a_ln_rate = 0.01
c_ln_rate = 0.1
model_ln_rate = 0.01
beta = 1 
rbl_weight = [1.5, 1.5]
ebl_weight = [0.1, 0.1]

## Peturbation:
targets = [-30, -20, -10, 0, 10, 20, 30]
y_star = torch.tensor(targets,dtype=torch.float32).unsqueeze(-1) * 0.0176

model = Mot_model()

actor = Actor(action_s=1, ln_rate = a_ln_rate, trainable = True) # 1D environment
estimated_model = Mot_model(ln_rate=model_ln_rate,lamb=None, Fixed = False)

CAG = CombActionGradient(actor, beta, rbl_weight, ebl_weight)

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
    trial_acc.append(torch.sqrt(rwd.detach()).mean().item())
    
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
    est_y = estimated_model.step(action)  # re-estimate values since model has been updated
    CAG.update(y, est_y, action, mu_a, std_a, delta_rwd)

    # Store variables after pre-train (including final trials without a perturbation)
    if ep % t_print ==0:
        accuracy = sum(trial_acc) / len(trial_acc)
        print("ep: ",ep)
        print("accuracy: ",accuracy)
        #print("std_a: ", std_a,"\n")
        tot_accuracy.append(accuracy)
        trial_acc = []

#print("agent mu weight: ", actor.l1.weight.item())
#print("agent mu bias: ", actor.l1.bias.item())
#print("agent std weight: ", actor.l2.weight.item())
#print("agent std bias: ", actor.l2.bias.item())

## ===== Save results =========
# Create directory to store results
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'results/model')
# Create directory if it did't exist before
os.makedirs(file_dir, exist_ok=True)

# Store model
if beta ==0:
    data = 'RBL_model.pt'
elif beta ==1:    
    data = 'EBL_model.pt'
else:
    data = 'Mixed_model.pt'
model_dir = os.path.join(file_dir,data)

if save_file:
    torch.save({
        "Targets": targets,
        'Actor': actor.state_dict(),
        'Net_optim': actor.optimiser.state_dict(),
        'Mean_rwd': mean_rwd,
        'Est_model': estimated_model.state_dict(),
        'Model_optim': estimated_model.optimiser.state_dict(),
    }, model_dir)
