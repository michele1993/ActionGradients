import os
from NN_motor_model  import Mot_model
from NN_Agent import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from NN_CombinedAG import CombActionGradient

''' Check what beta value gives best accuracy across different levels of noise, by training an agent from scratch'''

seeds = [3009, 5467, 1092, 9372,2801]

trials = 5000
t_print = 250
save_file = False

## set dimension
action_s = 10
output_s = 3
n_targets = 10

# Set noise variables
sensory_noises = torch.linspace(0,0.5,10)
fixd_a_noise = torch.tensor([0]).unsqueeze(-1) # set to experimental data value

# Set update variables
a_ln_rate = 0.001
c_ln_rate = 0.1
model_ln_rate = 0.001
betas = [0, 0.25, 0.5, 0.75, 1] #np.arange(0,11,2) /10.0 # [0,0.5,1] 

## Peturbation:
cue = torch.eye(n_targets,n_targets)

seed_best_betas = []
for s in seeds:
    torch.manual_seed(s)
    np.random.seed(s)
    best_betas = []

    model = Mot_model(action_s=action_s,output_s=output_s)
    y_star = torch.clip(torch.randn(n_targets,output_s),-1,1)

    for noise in sensory_noises:
        beta_accuracy = []

        for b in betas:
            # Initialise differento components
            actor = Actor(input_s=n_targets, action_s=action_s, ln_rate = a_ln_rate)
            estimated_model = Mot_model(action_s=action_s,output_s=output_s,ln_rate=model_ln_rate, Fixed=False)
            CAG = CombActionGradient(actor, b)

            mean_rwd = 0
            trial_acc = []

            for ep in range(1,trials+1):
                # Sample action from Gaussian policy
                action, mu_a, std_a = actor.computeAction(cue, fixd_a_noise)

                # Perform action in the env
                true_y = model.step(action.detach())
                
                # Add noise to sensory obs
                y = true_y + torch.randn_like(true_y) * noise

                # Compute differentiable rwd signal
                y.requires_grad_(True)
                rwd = torch.mean((y - y_star)**2, dim=-1,keepdim=True) # it is actually a punishment
                true_rwd = torch.mean((true_y - y_star)**2,dim=-1,keepdim=True) # it is actually a punishment
                trial_acc.append(torch.mean(torch.sqrt((true_y - y_star)**2),dim=-1).detach().mean().item())
                
                ## ====== Use running average to compute RPE =======
                #delta_rwd = rwd - mean_rwd
                delta_rwd = true_rwd - mean_rwd
                mean_rwd += c_ln_rate * delta_rwd.detach()
                ## ==============================================

                # Update the model
                est_y = estimated_model.step(action.detach())
                model_loss = estimated_model.update(y, est_y)

                # Update actor based on combined action gradient
                est_y = estimated_model.step(action)  # re-estimate values since model has been updated

                ## ---- Compute mixed action gradient ----
                # (do it manually so that can use true_rwd)
                R_grad = CAG.computeRBLGrad(action, mu_a, std_a, delta_rwd)
                E_grad = CAG.computeEBLGrad(y, est_y, action, mu_a, std_a, rwd)

                R_grad_norm = torch.norm(R_grad, dim=-1, keepdim=True) + 1e-12
                E_grad_norm = torch.norm(E_grad, dim=-1, keepdim=True) + 1e-12

                # Combine the two gradients angles
                comb_action_grad = b * E_grad/E_grad_norm + (1-b) * R_grad/R_grad_norm 

                # Combine the two gradients norms
                comb_action_grad *= b * E_grad_norm + (1-b) * R_grad_norm

                action_variables = torch.cat([mu_a,std_a],dim=-1)

                # ----- Update actor ----
                actor.ActionGrad_update(comb_action_grad, action_variables)

                # Store variables after pre-train (including final trials without a perturbation)
                if ep % t_print ==0:
                    accuracy = sum(trial_acc) / len(trial_acc)
                    #print('ep: ',ep, 'beta: ',b, 'accuracy: ',accuracy)
                    trial_acc = []

            beta_accuracy.append(accuracy) # store final accuracy

        beta_accuracy = np.array(beta_accuracy)
        best_betas.append(betas[np.argmin(beta_accuracy)])
        #print(beta_accuracy)
        print("Noise level: ",noise, "Best beta: ", best_betas[-1],'\n') 
    seed_best_betas.append(np.array(best_betas))

## ===== Save results =========
# Create directory to store results
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'results','Noisy_Forward',str(s))
# Create directory if it did't exist before
os.makedirs(file_dir, exist_ok=True)

# Store model
data = 'Noise_'+str(round(noise.item(),3))+'model.pt'
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
