import sys
sys.path.append('..')
import os
from Linear_motor_model  import Mot_model
from Agent import *
import torch
import numpy as np
from CombinedAG import CombActionGradient


""" Check how sum of a trained RBL poliy performs when sensory info becames available (suddenly or gradually) introducing the contribution of a trained EBL policy""" 

seeds = [8271, 1841, 5631, 9621, 8501]
initial_beta = 0 # load a policy that is assumed to be trained with RBL (i.e., only rewards available)
sudden = True # sudden change in visual feedback
trials = 1000
initial_trials = 200
t_print = 10
save_file = False

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

# Load models
if initial_beta ==0:
    data = 'RBL_'
elif initial_beta ==1:    
    data = 'EBL_'
else:
    data = 'Mixed_'


file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'results') # For the mixed model
# Load pretrained agent from 1st seed only
model_dir = os.path.join(file_dir,str(0),data+'model.pt') # For the mixed model
models = torch.load(model_dir)

tot_accuracy = []
for s in seeds:
    torch.manual_seed(s)
    np.random.seed(s)
    model = Mot_model()

    ebl_actor = Actor(ln_rate = a_ln_rate, trainable = True)
    ebl_actor.load_state_dict(models['ebl_actor'])

    rbl_actor = Actor(ln_rate = a_ln_rate, trainable = True)
    rbl_actor.load_state_dict(models['rbl_actor'])

    CAG = CombActionGradient(None,initial_beta)

    mean_rwd = 0
    trial_acc = []
    ep_acc = [] 
    beta = initial_beta
    for ep in range(1,trials+1):

        if sudden:
            if ep > initial_trials:
                beta = 1
        elif ep % initial_trials==0:
            beta+=0.25

        # Sample action from Gaussian policy
        rbl_a, rbl_mu, rbl_std = rbl_actor.computeAction(y_star, fixd_a_noise/2)
        ebl_a, ebl_mu, ebl_std = ebl_actor.computeAction(y_star, fixd_a_noise/2)

        action = beta * ebl_a + (1-beta) * rbl_a

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
        if ep > initial_trials:
            rbl_grad = CAG.computeRBLGrad(rbl_a,rbl_mu,rbl_std,delta_rwd)
            ebl_grad = CAG.computeEBLGrad(y,y,action,ebl_mu,ebl_std, rwd)

            rbl_a_variables = torch.cat([rbl_mu, rbl_std],dim=-1)
            rbl_actor.ActionGrad_update(rbl_grad, rbl_a_variables)

            ebl_a_variables = torch.cat([ebl_mu, ebl_std],dim=-1)
            ebl_actor.ActionGrad_update(ebl_grad, ebl_a_variables)

        if ep % t_print ==0:
            ep_acc.append(sum(trial_acc)/len(trial_acc))
            trial_acc = []

    # Store Accuracy for each seed
    tot_accuracy.append(ep_acc)
    ep_acc = []

tot_accuracy = np.array(tot_accuracy)

## ===== Save results =========
# Create directory to store results
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'results')

# Store model
if sudden:
    data = 'Sudden_ebl_feedback_'
else:
    data = 'Change_in_feedback_'

acc_dir = os.path.join(file_dir,data+'accuracy.npy')

if save_file:
    # Create directory if it did't exist before
    os.makedirs(file_dir, exist_ok=True)
    np.save(acc_dir,tot_accuracy)
