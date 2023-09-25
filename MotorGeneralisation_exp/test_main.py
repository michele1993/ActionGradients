import os
from Linear_motor_model  import Mot_model
from Agent import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from CombinedAG import CombActionGradient

# Set noise variables
sensory_noise = 0.01
fixd_a_noise = 0.02 # set to experimental data value

# Set update variables
a_ln_rate = 0.01
c_ln_rate = 0.1
model_ln_rate = 0.01
rbl_weight = [1.5, 1.5]
ebl_weight = [0.1, 0.1]

## Peturbation:
test_targets = [-45 -35 -25, -15, -5, 5, 15, 25, 35 ,45]
y_star = torch.tensor(test_targets,dtype=torch.float32).unsqueeze(-1) * 0.0176

beta = 0 
if beta == 0:
    label = "RBL"
elif beta == 1:        
    label = "EBL"
else:        
    label = "Mixed"


# Load models
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'results') # For the mixed model
model_dir = os.path.join(file_dir,'model',label+'_model.pt') # For the mixed model
models = torch.load(model_dir)

actor = Actor(action_s=1, ln_rate = a_ln_rate, trainable = True) # 1D environment
actor.load_state_dict(models['Actor'])

# Initialise additional components
model = Mot_model()
CAG = CombActionGradient(actor=actor, beta=beta, rbl_weight=rbl_weight, ebl_weight=ebl_weight)

action, mu_a, std_a = actor.computeAction(y_star, fixd_a_noise)

# Perform action in the env
true_y = model.step(action.detach())

rwd = (true_y - y_star)**2 # it is actually a punishment

print(rwd.mean())
exit()


## Save data
if beta == 0:
    label = "RBL"
elif beta == 1:        
    label = "EBL"
else:        
    label = "Mixed"
outcome_dir = os.path.join(file_dir,label+'_outcome_variability') # For the mixed model

# Save all outcomes so that can then plot whatever you want
if save_file: 
    np.save(outcome_dir, tot_outcomes)

## Plot actions:
t = np.arange(1,len(tot_outcomes)+1)
plt.plot(t,tot_outcomes)
plt.show()
