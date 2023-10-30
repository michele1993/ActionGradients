import os
from Linear_motor_model  import Mot_model
from Agent import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from CombinedAG import CombActionGradient

''' Code to demonostrate correlation between Forward model accuracy and policy generalisation error '''

torch.manual_seed(0)
seeds = [8721, 5467, 1092, 9372,2801]
save = True
# Set noise variables
sensory_noises = torch.linspace(0.01,0.25,10)
fixd_a_noise = 0.02 # set to experimental data value

# Set update variables
a_ln_rate = 0.01
model_ln_rate = 0.01
beta = 1

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
seed_forward_acc = []
for s in seeds:
    noise_acc = []
    forward_noise_acc = []
    for noise in sensory_noises:

        file_dir = os.path.dirname(os.path.abspath(__file__))
        file_dir = os.path.join(file_dir,'results','Noisy_Forward',str(s))
        data = 'Noise_'+str(round(noise.item(),3))+'model.pt'
        model_dir = os.path.join(file_dir,data)
        models = torch.load(model_dir)

        # Initialise Actor
        actor = Actor(action_s=1, ln_rate = a_ln_rate, trainable = True) # 1D environment
        actor.load_state_dict(models['Actor'])

        # Initialise FF model
        estimated_model = Mot_model(ln_rate=model_ln_rate,lamb=None, Fixed = False)
        estimated_model.load_state_dict(models['Est_model'])

        action, mu_a, std_a = actor.computeAction(y_star, fixd_a_noise)

        # Perform action in the env
        true_y = model.step(action.detach())
        est_y = estimated_model.step(action.detach())

        forward_error = (true_y - est_y.detach())**2
        forward_noise_acc.append(forward_error.mean())

        rwd = np.sqrt((true_y - y_star)**2) # it is actually a punishment
        noise_acc.append(rwd.mean())

    seed_acc.append(np.array(noise_acc))
    seed_forward_acc.append(np.array(forward_noise_acc))

seed_acc = np.array(seed_acc)
seed_forward_acc = np.array(seed_forward_acc)


## Select mean and std for corresponding values
mean_seed_acc = seed_acc.mean(axis=0)
std_seed_acc = seed_acc.std(axis=0)

mean_seed_forward_acc = seed_forward_acc.mean(axis=0)
std_seed_forward_acc = seed_forward_acc.std(axis=0)

print(mean_seed_acc)
print(std_seed_acc)
#print(mean_seed_forward_acc)
exit()

## Save results
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'..','results','generalisation')
os.makedirs(file_dir, exist_ok=True)
outcome_dir = os.path.join(file_dir,'gener_results')
#if save:
#    np.save(outcome_dir,tot_outcomes)
