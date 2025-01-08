import os
from Linear_motor_model  import Mot_model
from Agent import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from CombinedAG import CombActionGradient
import json

""" 
Test agent ability to learn the correct reaching angle based on binary success/failure feedback only, while varying DA levels.
This is equivalent to a reaching task where no visual target is provided and ppts need to explore the reaching space
only getting a success/failure signal whenever they reach within a certain radius of th target.
"""

seeds = [8721, 5467, 1092, 9372,2801]
save_file = False

tot_trials = 10000
t_print = 100
rwd_area = 0.087 # < 5 degree error allowed
target = 0.35 # target angle : 20 degrees 
y_star = torch.tensor([target],dtype=torch.float32)

# Set noise variables
sensory_noise = 0.01
fixd_a_noise = 0.02# set to experimental data value

# Set update variables
a_ln_rate = 0.0005
beta = 0.5
model_ln_rate = 0.01

if beta >0:
    CB_label = f'CB_{beta}_contribution_'
else:
    CB_label=''

# Initialise useful components
CAG = CombActionGradient(actor=None, beta=beta)
model = Mot_model()

DA_reduction = [1,0.1, 0.01, 0.001]
i=0
for da in DA_reduction:
    seed_acc = []
    seed_mu_a = []
    seed_std_a = []
    for s in seeds:
        torch.manual_seed(s)

        ## Reinitialise all the models for each run with a different beta
        actor = Actor(ln_rate = a_ln_rate, trainable = True, opt_type='SGD')
        estimated_model = Mot_model(ln_rate=model_ln_rate,lamb=None,Fixed=False)

        tot_accuracy = []
        tot_std = []
        tot_mu = []
        # Select an arbitrary non-zero input since target unknown
        policy_target = torch.ones(1) 
        successes = 0

        for ep in range(0,tot_trials+1):

            # pass random target since don't know where it is
            # Sample action from Gaussian policy
            action, mu_a, std_a = actor.computeAction(policy_target, fixd_a_noise)

            tot_mu.append(mu_a.detach().item())
            tot_std.append(std_a.detach().item())

            # Perform action in the env
            true_y = model.step(action.detach())
            
            # Add noise to sensory obs
            y = true_y + torch.randn_like(true_y) * sensory_noise 

            # Update the model
            est_y = estimated_model.step(action.detach())
            model_loss = estimated_model.update(y, est_y)

            # Compute differentiable error signal
            y.requires_grad_(True)
            error = 0.1*(y - y_star)**2 # it is actually a punishment

            ## Give a rwd if reach is within target area
            if torch.sqrt(error) <= rwd_area:
                rwd = 1
                successes +=1
            else:
                rwd = -1

            rwd *= da

            # Update actor based on RBL action gradient only since binary feedback
            RBL_grad =  -1*CAG.computeRBLGrad(action, mu_a, std_a, rwd) #NOTE: Need -1 since now rwd is a rwd we want to max and not an error/distance we want to min!!!

            est_y = estimated_model.step(action)  # re-estimate values since model has been updated
            EBL_grad =  CAG.computeEBLGrad(y, est_y, action, mu_a, std_a, error) 

            action_gradient = beta * EBL_grad + (1-beta) * RBL_grad

            action_variables = torch.cat([mu_a, std_a],dim=-1)


            actor.ActionGrad_update(action_gradient, action_variables)

            # Store variables 
            accuracy = np.sqrt(error.detach().item())
            if ep % t_print ==0:
                accuracy = successes / t_print
                tot_accuracy.append(accuracy)
                print('DA reduction: ', da)
                print("ep: ",ep)
                print("accuracy: ",successes)
                #print("mu: ", mu_a)
                #print("std: ", std_a, '\n')
                successes = 0
            
        seed_acc.append(tot_accuracy)
        seed_mu_a.append(tot_mu)
        seed_std_a.append(tot_std)

    results = {
        'accuracy': seed_acc,
        'mu_a': seed_mu_a,
        'std_a': seed_std_a
    }
    label = CB_label+"DA_decrease_"+str(i)+'.json'
    i+=1
    root_dir = os.path.dirname(os.path.abspath(__file__))
    file_dir = os.path.join(root_dir,'results') # For the mixed model
    os.makedirs(file_dir, exist_ok=True)
    result_dir = os.path.join(file_dir,label) # For the mixed model

    # Save all outcomes so that can then plot whatever you want
    if save_file: 
        # Convert and write JSON object to file
        with open(result_dir, "w") as outfile: 
            json.dump(results, outfile) 
