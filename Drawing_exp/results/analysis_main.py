import sys
import argparse
sys.path.append('..')
import os
from Kinematic_Motor_model  import Kinematic_model
from Forward_model import ForwardModel
from rnn_actor import Actor
import torch
import numpy as np
import matplotlib.pyplot as plt
from CombinedAG import CombActionGradient
from utils import compute_targetLines

#torch.manual_seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('--step_x_update','-sxu', type=int, nargs='?', default=1)

## Argparse variables:
args = parser.parse_args()

step_x_update = args.step_x_update

if step_x_update == 1:
    seeds = [4634, 5637, 9920, 2310, 7169]
else:
    ## Last seed of above was exluced due to massime outliers on all conditions
    seeds = [4634, 5637, 9920, 2310, 3021]

betas = [0,0.5,1]

save_results = True
action_s = 2 # two angles in 2D kinematic arm model
state_s = 2 # 2D space x,y-coord
a_ln_rate = 0

# Set noise variables
sensory_noise = 0#.01
fixd_a_noise = 0.02 # set to experimental data value

# Set experiment variables
n_target_lines = 6
n_steps = 10


# Initialise env
model = Kinematic_model()

tot_accuracy = []
tot_acc_std = []
tot_trajects = []
tot_RBL_grad = []
tot_EBL_grad = []
for b in betas:

    if b == 0:
        data = 'RBL'
    elif b == 1:    
        data = 'EBL'
    else:
        data = 'Mixed_'+str(b)

    seed_acc = []
    for s in seeds:
        # ==== Initialise components ==========
        actor = Actor(input_s= n_target_lines, batch_size=n_target_lines, ln_rate = a_ln_rate, learn_std=True)

        # Load Agent 
        file_dir = os.path.dirname(os.path.abspath(__file__))
        file_dir = os.path.join(file_dir,str(step_x_update)+'_update')
        seed_dir = os.path.join(file_dir,str(s))

        model_dir = os.path.join(seed_dir,data+'_model.pt')

        models = torch.load(model_dir)
        actor.load_state_dict(models['Actor'])
        actor.optimizer.load_state_dict(models['Net_optim'])

        origin = models['Origin']
        targets = models['Targets']

        x_targ = targets[0]
        y_targ = targets[1]

        trial_acc = []
        outcomes = []

        outcomes.append(origin.numpy())

        # Initialise cues at start of each trial
        cue = torch.eye(n_target_lines).unsqueeze(0) # each one-hot denotes different cue

        for t in range(n_steps):
            # Sample action from Gaussian policy
            action, mu_a, std_a = actor.computeAction(cue, fixd_a_noise)
            action = mu_a

            # Perform action in the env
            true_x_coord,true_y_coord = model.step(action.detach())

            # Add noise to sensory obs
            x_coord = true_x_coord + torch.randn_like(true_x_coord) * sensory_noise 
            y_coord = true_y_coord + torch.randn_like(true_y_coord) * sensory_noise 

            # Compute differentiable rwd signal
            coord = torch.cat([x_coord,y_coord], dim=1)
            outcomes.append(coord.numpy())

            rwd = torch.sqrt((coord[:,0:1] - x_targ[:,t:t+1])**2 + (coord[:,1:2] - y_targ[:,t:t+1])**2) # it is actually a punishment
            trial_acc.append(rwd.detach().mean().item())
            
            current_x = x_coord
            current_y = y_coord
            cue = torch.randn_like(cue) # each one-hot denotes different cue

        # Store accuracy across each seed
        seed_acc.append(sum(trial_acc) /len(trial_acc))

        ## Store trajectories for plotting based only on 1 seed
        ## averaging over seeds wouldn't be fair
        if s ==9920:
            tot_trajects.append(outcomes)

        # Store gradient norm for mixed model across seeds
        if b == 0.5:
            RBL_grad, EBL_grad = np.load(os.path.join(seed_dir, 'Mixed_0.5_gradients.npy'))
            tot_RBL_grad.append(RBL_grad)
            tot_EBL_grad.append(EBL_grad)
    
    # Store accuracy for each beta value
    tot_accuracy.append(np.mean(seed_acc))
    tot_acc_std.append(np.std(seed_acc))

    
tot_RBL_grad = np.array(tot_RBL_grad).mean(axis=0)
tot_EBL_grad = np.array(tot_EBL_grad).mean(axis=0)

tot_grad = np.stack([tot_RBL_grad, tot_EBL_grad])
## ===== Analyse results =========
#print(sum(trial_acc)/len(trial_acc))
acc_dir = os.path.join(file_dir,'Ataxia_score.npy')
outcomes_dir = os.path.join(file_dir,'Traject_outcomes.npy')
grad_dir = os.path.join(file_dir,'MixedModel_grads.npy')

tot_data = np.stack([tot_accuracy,tot_acc_std])
outcomes = np.array(tot_trajects)

if save_results:
    np.save(acc_dir,tot_data)
    if step_x_update ==1:
        np.save(outcomes_dir,outcomes)
        np.save(grad_dir,tot_grad)
