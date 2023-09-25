from Motor_model  import Mot_model
from Agent import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from CombinedAG import CombActionGradient

 

torch.manual_seed(0)

pre_train = 3000
## Set trials to match Izawa and Shadmer, 2011 experimental set-up, where they add 1 degree pertubation every 40 trials up to 8 degreese
trials_x_perturbation = 40
n_pertubations = 8
perturb_trials = trials_x_perturbation * n_pertubations # 8 * 40 , n. perturb x trials_per_perturb
fixed_trials = 100 # final trials at 8 degree rotation pertub, based on which action variance is assessed
tot_trials = pre_train + perturb_trials + fixed_trials
perturbation_increase = 0.0176 # equivalent to 1 degree
max_perturbation = perturbation_increase * n_pertubations

# Set noise variables
sensory_noise = 0.01
fixd_a_noise = 0.02 # set to experimental data value

# Set update variables
a_ln_rate = 0.01
c_ln_rate = 0.1
model_ln_rate = 0.01
beta_mu = 0
beta_std = beta_mu
rbl_std_weight =  1.5
ebl_std_weight = 0.1

## Peturbation:

target = 0.1056 # target angle : 56 degrees 
y_star = torch.tensor([target],dtype=torch.float32)

model = Mot_model()

actor = Actor(output_s=2, ln_rate = a_ln_rate, trainable = True)
estimated_model = Mot_model(ln_rate=model_ln_rate,lamb=None, Fixed = False)

CAG = CombActionGradient(actor, beta_mu, beta_std, rbl_std_weight, ebl_std_weight)

tot_accuracy = []
tot_actions = []
tot_outcomes = []

mean_rwd = 0
# start with a zero perturbation
current_perturbation = 0

for ep in range(0,tot_trials):

    # Sample action from Gaussian policy
    action, mu_a, std_a = actor.computeAction(y_star, fixd_a_noise)

    # Perform action in the env
    true_y = model.step(action.detach())
    
    ## ====== Increase perturbation =======
    # Follow procedure from Izawa and Shadmer, 2011
    # after pre_train add a 1-degree perturbation every 40 trials upto 8 degree perturbation
    if ep >= pre_train and ep % trials_x_perturbation == 1 and current_perturbation < max_perturbation:
       current_perturbation -= perturbation_increase
       #print("\n Ep: ", ep, "Perturbation: ", current_perturbation, "\n")
    ## ==============================================

    # Add noise to sensory obs
    y = true_y + torch.randn_like(true_y) * sensory_noise 

    # Add current perturbation to observation
    y+= current_perturbation

    # Compute differentiable rwd signal
    y.requires_grad_(True)
    rwd = (y - y_star)**2 # it is actually a punishment
    
    ## ====== Use running average to compute RPE =======
    delta_rwd = rwd - mean_rwd
    mean_rwd += c_ln_rate * delta_rwd.detach()
    ## ==============================================

    # For rwd-base learning give rwd of 1 if reach better than previous else -1
    if beta_mu == 0:
       delta_rwd /= torch.abs(delta_rwd.detach()) 


    # Update the model
    est_y = estimated_model.step(action.detach())
    model_loss = estimated_model.update(y, est_y)

    # Update actor based on combined action gradient
    #if ep > pre_train:
    est_y = estimated_model.step(action)  # re-estimate values since model has been updated
    CAG.update(y, est_y, action, mu_a, std_a, delta_rwd)

    # Store variables after pre-train (including final trials without a perturbation)
    if ep >= (pre_train - trials_x_perturbation):
        accuracy = np.sqrt(rwd.detach().numpy())
        print("ep: ",ep)
        print("accuracy: ",accuracy)
        #print("std_a: ", std_a,"\n")
        tot_accuracy.append(accuracy)
        tot_actions.append(action.detach().numpy())
        tot_outcomes.append(true_y.detach().numpy() - target) # make it so that 0 radiants refer to the target

print("Tot variability: ",np.std(tot_outcomes[-fixed_trials:]))

## Plot actions:
t = np.arange(1,len(tot_outcomes)+1)
plt.plot(t,tot_outcomes)
plt.show()
