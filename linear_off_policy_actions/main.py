import os
from Linear_motor_model  import Mot_model
from Agent import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from offPol_CombinedAG import CombActionGradient
from replayBuffer import MemoryBuffer


""" Generate a EBL or RBL policy with decaying weights, then show offline RBL allow consolidation of RBL and EBL by preventing weight decay"""

seeds = [47382, 87102, 71092, 19283, 32912]
online_learning_trials = 250
offline_trials = 250
tot_trials = online_learning_trials + offline_trials
offline_learning = True
t_print = 10
save_file = False
beta = 1
# Plot how long term retention due to RBL replay is affected by DA deficiencies
offline_DA_reduction = 1000

# Model file names for storing base on original beta value
if beta ==0:
    data = f'RBL_DA_x{offline_DA_reduction}_reduction_data'
elif beta ==1:    
    data = f'EBL_DA_x{offline_DA_reduction}_reduction_data'
else:
    data = f'Mixed_DA_x{offline_DA_reduction}_reduction_data'

if offline_learning:
    data = 'Offline_'+data
else:
    data = 'No_Offline_'+data


# Set noise variables
sensory_noise = 0.01
fixd_a_noise = 0.02 # set to experimental data value

# Set update variables
a_ln_rate = 0.005
c_ln_rate = 0.01
model_ln_rate = 0.01
buffer_size = 100

## Peturbation:

target = 8 * 0.1056 # random target angle : 36 degrees 
y_star = torch.tensor([target],dtype=torch.float32)

model = Mot_model()

seed_acc = []
print("\nBeta: ",beta)
for s in seeds:

    # Set random seeds
    torch.manual_seed(s)
    np.random.seed(s)

    # Initialise components
    actor = Actor(ln_rate = a_ln_rate, opt_type='SGD')
    CAG = CombActionGradient(actor, beta)
    estimated_model = Mot_model(ln_rate=model_ln_rate,lamb=None, Fixed = False)
    buffer = MemoryBuffer(size=buffer_size)

    tot_accuracy = []
    mean_rwd = 0
    trial_acc = []

    #print("\n New seed: \n")

    for ep in range(1,tot_trials+1):

        # Sample action from Gaussian policy
        action, mu_a, std_a, p_action = actor.computeAction(y_star, fixd_a_noise)

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

        # Store data in buffer
        buffer.store_transition(y_star, action.detach(), delta_rwd.detach(), p_action.detach())

        # Update the model
        est_y = estimated_model.step(action.detach())
        model_loss = estimated_model.update(y, est_y)

        # Update actor based on combined action gradient
        est_y = estimated_model.step(action)  # re-estimate values since model has been updated

        ## ----- Compute two action gradients ----
        if ep < online_learning_trials:
            # Learn only based on the desired online action gradients (controlled by \beta)
            R_grad = CAG.computeRBLGrad(action, mu_a, std_a, delta_rwd)
            E_grad = CAG.computeEBLGrad(y, est_y, action, mu_a, std_a, delta_rwd)

        # At the end of learning trials set EBL gradient to zero, preventing any online EBL learning to occur
        # with weight decay briging weight back to zero, unless RBL offline learning kicks in
        else:
            # Set beta to zero since only the RBL grad can be used offline
            beta = 0
            # Set both gradient to zero for precaution to prevent any learning
            # other than the offline RBL grad computed below when offline_learning=True
            E_grad = torch.tensor([0,0])
            R_grad = torch.tensor([0,0])
            
            # Compute offline RBL action gradient if offline learning taking place
            if offline_learning:
                # Sample from memory buffer
                y_star, action, delta_rwd, old_p_action = buffer.sample_transition()
                # DA deficits
                delta_rwd /= offline_DA_reduction
                # COmpute mean based on new (offline update) policy
                _, mu_a, std_a, _ = actor.computeAction(y_star,fixd_a_noise)
                # COmpute new prob. for old action based on new (offline update) policy
                new_p_action = actor.compute_p(action=action, mu=mu_a, std=std_a).detach()
                # Compute the offpolicy RBL action gradient
                R_grad = CAG.compute_offlineRBLGrad(new_p=new_p_action, old_p=old_p_action, action=action, mu_a = mu_a, std_a=std_a, delta_rwd=delta_rwd)

        # Combine the two gradients 
        comb_action_grad = beta * E_grad + (1-beta) * R_grad 

        action_variables = torch.cat([mu_a, std_a],dim=-1)
        ## ----------------------------
        # Update the action
        agent_grad = actor.ActionGrad_update(comb_action_grad, action_variables)

        #if ep == online_learning_trials+1:
        #    print("\n Offline learning \n")

        # Store variables after pre-train (including final trials without a perturbation)
        if ep % t_print ==0:
            accuracy = sum(trial_acc) / len(trial_acc)
            #print("ep: ",ep)
            #print("accuracy: ",accuracy)
            tot_accuracy.append(accuracy)
            trial_acc = []

    seed_acc.append(tot_accuracy)                

seed_acc = np.array(seed_acc)

print('Online performance: ', seed_acc.mean(axis=0)[online_learning_trials//t_print])
print('Retention: ', seed_acc.mean(axis=0)[-1])

## ===== Save results =========
# Create directory to store results
direct = os.path.dirname(os.path.abspath(__file__))
direct = os.path.join(direct,'results')


file_dir = os.path.join(direct,data)

if save_file:
    # Create directory if it did't exist before
    os.makedirs(direct, exist_ok=True)
    np.save(file_dir, seed_acc)
