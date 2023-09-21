import sys
sys.path.append('..')
import os
from Motor_model  import Kinematic_model
from Forward_model import ForwardModel
from rnn_actor import Actor
import torch
import numpy as np
import matplotlib.pyplot as plt
from CombinedAG import CombActionGradient
from utils import compute_targetLines

torch.manual_seed(0)

save_file = False
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

# ==== Initialise components ==========
actor = Actor(input_s= n_target_lines, batch_size=n_target_lines, ln_rate = a_ln_rate, learn_std=True)

# Load Agent 
beta = 1
file_dir = os.path.dirname(os.path.abspath(__file__))

if beta ==0:
    data = 'RBL_model.pt'
elif beta ==1:    
    data = 'EBL_model.pt'
else:
    data = 'Mixed_model.pt'
model_dir = os.path.join(file_dir,'model',data)

models = torch.load(model_dir)
actor.load_state_dict(models['Actor'])
actor.optimizer.load_state_dict(models['Net_optim'])

origin = models['Origin']
targets = models['Targets']

acc = models['Accuracy']
print(acc[-1])

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
    trial_acc.append(rwd.detach().numpy())
    
    current_x = x_coord
    current_y = y_coord
    cue = torch.randn_like(cue) # each one-hot denotes different cue

## ===== Analyse results =========
outcomes = np.array(outcomes)
plt.plot(outcomes[:,:,0],outcomes[:,:,1])
plt.show()
