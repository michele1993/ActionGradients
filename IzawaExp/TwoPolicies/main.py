import sys
sys.path.append('..')
import os
from Linear_motor_model  import Mot_model
from Agent import *
import torch
import numpy as np
from CombinedAG import CombActionGradient

seeds = [0, 9284, 5992, 7861, 1594]

""" Train two separate policies whose outputs are summed, one with EBL and the other with RBL"""
use_beta = False # also test condition where there is no action weighting but simply summed to show this is worse than using beta=1 (i.e., need the beta)
beta = 1
trials = 5000
# NOTE: ensure pre-training for no_beta condition, to start at same accuracy as EBL
if not use_beta:
    trials = 5100 
t_print = 100
save_file = True

# Set noise variables
sensory_noise = 0.0001
fixd_a_noise = 0.0001 # set to experimental data value

# Set update variables
a_ln_rate = 0.01 
c_ln_rate = 0.1
model_ln_rate = 0.01

## Peturbation:

target = 0.1056 # target angle : 6 degrees - Izawa and Shadmer, 2011
y_star = torch.tensor([target],dtype=torch.float32)

model = Mot_model()

for s in seeds:
    torch.manual_seed(s)
    np.random.seed(s)

    ebl_actor = Actor(ln_rate = a_ln_rate, trainable = True)
    rbl_actor = Actor(ln_rate = a_ln_rate, trainable = True)

    CAG = CombActionGradient(None,beta)

    tot_accuracy = []
    mean_rwd = 0
    trial_acc = []
    for ep in range(0,trials):

        rbl_a, rbl_mu, rbl_std = rbl_actor.computeAction(y_star, fixd_a_noise/2)
        ebl_a, ebl_mu, ebl_std = ebl_actor.computeAction(y_star, fixd_a_noise/2)

        if use_beta:
            action = beta * ebl_a + (1-beta) * rbl_a
        else:
            action =  (ebl_a + rbl_a) 

        # Perform action in the env
        true_y = model.step(action)
        
        # Add noise to sensory obs
        y = true_y + torch.randn_like(true_y) * sensory_noise 

        # Compute differentiable rwd signal
        rwd = (y - y_star)**2 # it is actually a punishment
        trial_acc.append(torch.sqrt(rwd.detach()).item())
        
        ## ====== Use running average to compute RPE =======
        delta_rwd = rwd.detach() - mean_rwd
        mean_rwd += c_ln_rate * delta_rwd.detach()
        ## ==============================================


        # Update actor based on combined action gradient
        rbl_grad = CAG.computeRBLGrad(rbl_a,rbl_mu,rbl_std,delta_rwd)
        ebl_grad = CAG.computeEBLGrad(y,y,action,ebl_mu,ebl_std, rwd)


        rbl_a_variables = torch.cat([rbl_mu, rbl_std],dim=-1)
        rbl_actor.ActionGrad_update(rbl_grad, rbl_a_variables)

        ebl_a_variables = torch.cat([ebl_mu, ebl_std],dim=-1)
        ebl_actor.ActionGrad_update(ebl_grad, ebl_a_variables)

        # Store variables after pre-train (including final trials without a perturbation)
        if ep % t_print ==0:
            accuracy = sum(trial_acc) / len(trial_acc)
            print("ep: ",ep)
            print("accuracy: ",accuracy)
            tot_accuracy.append(accuracy)
            trial_acc = []

    ## ===== Save results =========
    # Create directory to store results
    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_dir = os.path.join(file_dir,'results',str(s))

    # Store model
    if beta ==0:
        data = 'RBL_'
        std_a = rbl_std
    elif beta ==1:
        data = 'EBL_'
        std_a = ebl_std
    else:
        data = 'Mixed_'+str(beta)+'_'
        std_a = ebl_std + rbl_std

    model_dir = os.path.join(file_dir,data+'model.pt')
    if save_file:
        # Create directory if it did't exist before
        os.makedirs(file_dir, exist_ok=True)
        if use_beta:
            torch.save({
                'beta': beta,
                'rbl_actor': rbl_actor.state_dict(),
                'ebl_actor': ebl_actor.state_dict(),
            }, model_dir)
        else:
            std_a = ebl_std + rbl_std
            data = 'NoBeta_'
        # Save accuracy
        acc_dir = os.path.join(file_dir,data+'accuracy.npy')
        std_dir = os.path.join(file_dir,data+'action_std.npy')
        np.save(acc_dir,tot_accuracy)
        np.save(std_dir,std_a.detach().numpy())
