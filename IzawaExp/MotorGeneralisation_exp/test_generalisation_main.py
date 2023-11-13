import os
from Linear_motor_model  import Mot_model
from Agent import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from CombinedAG import CombActionGradient

""" Load trained policy for each beta value and test generalisation performance """

torch.manual_seed(0)
seeds = [8721, 5467, 1092, 9372,2801]
save = True
# Set noise variables
sensory_noise = 0.01
fixd_a_noise = 0.02 # set to experimental data value

# Set update variables
a_ln_rate = 0.01
betas = np.arange(0,11,1) /10.0

## Generate N test targets between 38 and -38
# since in Izawa's test based on pertub upto 8 degrees (and training was on max 30 degrees)
N = 100
max_val, min_val = 30,-30
range_size = (max_val - min_val)  # 2
test_targets = np.random.rand(N) * range_size + min_val
#test_targets = [-45 -35 -25, -15, -5, 5, 15, 25, 35 ,45]
y_star = torch.tensor(test_targets,dtype=torch.float32).unsqueeze(-1) * 0.0176

model = Mot_model()
# Load models
seed_acc = []
for s in seeds:
    betas_acc = []
    for b in betas:

        file_dir = os.path.dirname(os.path.abspath(__file__))
        file_dir = os.path.join(file_dir,'results',str(s))
        model_file = 'Mixed_'+str(b)+'model.pt'
        model_dir = os.path.join(file_dir,model_file)
        models = torch.load(model_dir)

        actor = Actor(action_s=1, ln_rate = a_ln_rate, trainable = True) # 1D environment
        actor.load_state_dict(models['Actor'])

        action, mu_a, std_a = actor.computeAction(y_star, fixd_a_noise)

        # Perform action in the env
        true_y = model.step(action.detach())
        #true_y = model.step(mu_a.detach())

        rwd = np.sqrt((true_y - y_star)**2) # it is actually a punishment
        betas_acc.append(rwd.mean())

    seed_acc.append(np.array(betas_acc))

seed_acc = np.array(seed_acc)

## Select mean and std for corresponding values
mean_seed_acc = seed_acc.mean(axis=0)
std_seed_acc = seed_acc.std(axis=0)
## Select values for beta=0,0.3,1 for plotting purposes
# mixed with beta =0.3 since this is what we plotted for motor variab
# to match Izawa findings
RBL_mean = mean_seed_acc[0]
Mixed_mean = mean_seed_acc[3]
EBL_mean = mean_seed_acc[-1]
RBL_std = std_seed_acc[0]
Mixed_std = std_seed_acc[3]
EBL_std = std_seed_acc[-1]

## Save results
tot_outcomes = [[RBL_mean,Mixed_mean,EBL_mean],[RBL_std,Mixed_std, EBL_std]]
#tot_outcomes = [[mean_seed_acc],[std_seed_acc]]
tot_outcomes = np.array(tot_outcomes)
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'..','results','generalisation')
os.makedirs(file_dir, exist_ok=True)
outcome_dir = os.path.join(file_dir,'gener_results')
if save:
    np.save(outcome_dir,tot_outcomes)
