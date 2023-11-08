import os
import sys
import argparse
from Kinematic_Motor_model  import Kinematic_model
from Forward_model import ForwardModel
from Gradient_model import GradientModel
from actor import Actor
import torch
import numpy as np
import matplotlib.pyplot as plt
from CombinedAG import CombActionGradient
from utils import compute_targetLines
import matplotlib as mpl



# Train a model with mixed grad
beta = 0.5

    
torch.manual_seed(94747)
np.random.seed(94747)
save_file = False
n_trials = 2500
model_pretrain = 100
grad_pretrain = model_pretrain * 1
t_print = 100
action_s = 2 # two angles in 2D kinematic arm model
state_s = 2 # 2D space x,y-coord

# Set noise variables
sensory_noise = 0.01 #0.1 #0.0001
fixd_a_noise = 0.001 #.0002 # set to experimental data value

# Set update variables
assert beta >= 0 and beta <= 1, "beta must be between 0 and 1 (inclusive)"
gradModel_lr_decay = 1
actor_lr_decay = 1
a_ln_rate = 0.001
c_ln_rate = 0.05
model_ln_rate = 0.001
grad_model_ln_rate = 0.001
rbl_weight = [1,1]
ebl_weight = [1,1]

# Set experiment variables needed to define targets 
n_target_lines = 6
n_steps = 1


# Initialise env
model = Kinematic_model()
large_circle_radium = model.l1 + model.l2 # maximum reacheable coordinate based on lenght of entire arm 
small_circle_radius = model.l1 # circle defined by radius of upper arm

## ====== Generate a target for each cue (i.e. points) by taking final point on 6 lines ====== 

# Always start from the same initial origin point (x_0,y_0)
# Compute origin as point tanget to shoulder in the middle of reaching space (arbitary choice)
x_0 = 0
y_0 = (large_circle_radium - small_circle_radius)/2 + small_circle_radius
target_origin = [x_0,y_0]
distance_from_target = 0.2

## ----- Check that target line length do not go outside reacheable space -----
distance = np.sqrt(x_0**2 + y_0**2) # assuming shoulder is at (0,0)

## Bypass safety check as may be too stringent
assert (distance_from_target + distance) < large_circle_radium  and (distance-distance_from_target) > small_circle_radius, "line_lenght needs to be shorter or risk of going outside reacheable space"  

# Create single target by taking final point on a line
x_targ, y_targ = compute_targetLines(target_origin, n_target_lines, n_steps, distance_from_target) # shape: [n_target_lines, n_steps] , allowing batch training
x_targ = x_targ[:,-1:]
y_targ = y_targ[:,-1:]


## ==== Initialise components ==========
estimated_model = ForwardModel(state_s=state_s,action_s=action_s, max_coord=large_circle_radium, ln_rate=model_ln_rate)
grad_estimator = GradientModel(state_s=state_s,action_s=action_s, ln_rate=grad_model_ln_rate, lr_decay= gradModel_lr_decay)
actor = Actor(input_s= n_target_lines, ln_rate = a_ln_rate, learn_std=True,lr_decay=actor_lr_decay)
CAG = CombActionGradient(actor, action_s, rbl_weight, ebl_weight)


command_line = f'fixd_a_noise: {fixd_a_noise}, sensory_noise: {sensory_noise}, a_ln: {a_ln_rate}, rbl_weight: {rbl_weight}, ebl_weight: {ebl_weight}' 

tot_accuracy = []
mean_rwd = torch.zeros(n_target_lines,1)
trial_acc = []
model_losses = []

norm_ebl_gradients = []
norm_rbl_gradients = []
norm_EBL_tot_grad = []
norm_RBL_tot_grad = []

ebl_mu_gradients = []
rbl_mu_gradients = []
EBL_mu_tot_grad = []
RBL_mu_tot_grad = []

## ====== Diagnostic variables ========
grad_model_loss = []
tot_grad_model_loss = []
target_ebl_grads = []
tot_target_ebl_grads = []
## ===================================

# Initialise cues at start of each trial
cue = torch.eye(n_target_lines)

# Initialise starting position for each target line (start all from the same point)
current_x = torch.tensor([x_0]).repeat(n_target_lines,1)
current_y = torch.tensor([y_0]).repeat(n_target_lines,1)

for t in range(1,n_trials+1):

    # Sample action from Gaussian policy
    action, mu_a, std_a = actor.computeAction(cue, fixd_a_noise)

    # Perform action in the env
    true_x_coord,true_y_coord = model.step(action)

    # Add noise to sensory obs
    x_coord = true_x_coord.detach() + torch.randn_like(true_x_coord) * sensory_noise 
    y_coord = true_y_coord.detach() + torch.randn_like(true_y_coord) * sensory_noise 

    # Compute differentiable rwd signal
    coord = torch.cat([x_coord,y_coord], dim=1) 
    coord.requires_grad_(True)

    rwd = (coord[:,0:1] - x_targ)**2 + (coord[:,1:2] - y_targ)**2 # it is actually a punishment
    true_rwd = (true_x_coord - x_targ)**2 + (true_y_coord - y_targ)**2 # it is actually a punishment

    trial_acc.append(torch.mean(torch.sqrt(true_rwd.detach())).item())
    
    ## ====== Use running average to compute RPE =======
    delta_rwd = rwd - mean_rwd 
    mean_rwd += c_ln_rate * delta_rwd.detach()
    ## ==============================================

    # Update the model
    state_action = torch.cat([current_x,current_y,action],dim=1)
    est_coord = estimated_model.step(state_action.detach())
    model_loss = estimated_model.update(x_coord.detach(), y_coord.detach(), est_coord)
    model_losses.append(model_loss.detach()/n_steps)

    if t > model_pretrain: 
        # Compute gradients 
        est_coord = estimated_model.step(state_action)  # re-estimate values since model has been updated
        R_grad = CAG.computeRBLGrad(action, mu_a, std_a, delta_rwd)
        E_grad = CAG.computeEBLGrad(y=coord, est_y=est_coord, action=action, mu_a=mu_a, std_a=std_a, delta_rwd=delta_rwd)

        # Learn the EBL grad
        c_target = torch.cat([x_targ, y_targ],dim=-1).detach() 
        est_E_grad = grad_estimator(state_action.detach(),c_target)
        grad_loss = grad_estimator.update(E_grad, est_E_grad)
        grad_model_loss.append(grad_loss.detach())

        # Use the estimated gradient for training Actor
        E_grad = est_E_grad.detach() 

        # Store gradients
        if t > grad_pretrain:
            #gradient = beta * torch.clip(E_grad,-5,5) + (1-beta) * torch.clip(R_grad,-5,5)
            gradient = beta * E_grad + (1-beta) * R_grad
            a_variab = torch.cat([mu_a,std_a],dim=1) 

            actor.ActionGrad_update(gradient, a_variab)
            
            # Store the gradient magnitude for plotting purposes
            norm_ebl_gradients.append(torch.norm(E_grad,dim=-1))
            norm_rbl_gradients.append(torch.norm(R_grad, dim=-1))

            # Store the gradient of policy mean  for plotting purposes
            ebl_mu_gradients.append(torch.mean(E_grad[:,0:2],dim=0))
            rbl_mu_gradients.append(torch.mean(R_grad[:,0:2], dim=0))


    # Store variables after pre-train (including final trials without a perturbation)
    if t % t_print ==0:
        accuracy = sum(trial_acc) / len(trial_acc)
        m_loss = sum(model_losses) / len(model_losses)
        print("Trial: ",t)
        print("Accuracy: ",accuracy)
        #print("std_a: ", std_a)
        #print("Model Loss: ", m_loss)
        tot_accuracy.append(accuracy)
        trial_acc = []
        model_losses = []

## ===== Save results =========
# Create directory to store results
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'results')

# Store model
if beta ==0:
    data = 'RBL'
elif beta ==1:    
    data = 'EBL'
else:
    data = 'Mixed_'+str(beta)

model_dir = os.path.join(file_dir,data+'_model.pt')
current_x = torch.tensor([x_0]).repeat(n_target_lines,1)
current_y = torch.tensor([y_0]).repeat(n_target_lines,1)
origin = torch.cat([current_x,current_y],dim=1)

if save_file:
    os.makedirs(file_dir, exist_ok=True)
    # Create directory if it did't exist before
    # Store command line
    with open(os.path.join(file_dir,'commands.txt'), 'w') as f:
        f.write(command_line)
    torch.save({
        'Accuracy': tot_accuracy,
        'Origin': origin,
        'Targets': torch.stack([x_targ,y_targ]),
        'Actor': actor.state_dict(),
        'Net_optim': actor.optimizer.state_dict(),
        'Est_model': estimated_model.state_dict(),
        'Model_optim': estimated_model.optimiser.state_dict(),
        'Grad_model': grad_estimator.state_dict(),
        'Grad_model_optim': grad_estimator.optimiser.state_dict(),
    }, model_dir)
