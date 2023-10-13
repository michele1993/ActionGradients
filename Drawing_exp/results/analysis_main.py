import sys
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

torch.manual_seed(0)

save_results = False
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
step_x_update = 1

file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,str(step_x_update)+'_update')

if beta ==0:
    data = 'RBL'
elif beta ==1:    
    data = 'EBL'
else:
    data = 'Mixed'

model_dir = os.path.join(file_dir,'model',data+'_model.pt')

models = torch.load(model_dir)
actor.load_state_dict(models['Actor'])
actor.optimizer.load_state_dict(models['Net_optim'])

origin = models['Origin']
targets = models['Targets']

acc = models['Accuracy']
print('Final training accuracy: ',acc[-1],'\n')

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

## ===== Analyse results =========
#print(sum(trial_acc)/len(trial_acc))
acc_dir = os.path.join(file_dir,data+'_ataxia_score.npy')
outcomes_dir = os.path.join(file_dir,data+'_outcomes.npy')
outcomes = np.array(outcomes)

if save_results:
    np.save(acc_dir,np.array(trial_acc))
    np.save(outcomes_dir,np.array(outcomes))

plt.plot(outcomes[:,:,0],outcomes[:,:,1])
plt.show()
