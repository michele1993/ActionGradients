import sys
sys.path.append('/Users/px19783/code_repository/cerebellum_project/ActionGradients')
from Motor_model  import Mot_model
from Agent import *
import torch
import numpy as np
from CombinedAG import CombActionGradient

 

torch.manual_seed(0)

episodes = 2400 # 2000: pre-train, 320: perturb
a_ln_rate = 0.01
c_ln_rate = 0.1
model_ln_rate = 0.01
pre_train = 200
sensory_noise = 0.01
fixd_a_noise = 0.025#0.15
beta_mu = 0
beta_std = beta_mu
perturbation = 0
perturbation_increase = 0.0176


y_star = torch.ones(1)

model = Mot_model()

actor = Actor(output_s=2, ln_rate = a_ln_rate, trainable = True)
estimated_model = Mot_model(ln_rate=model_ln_rate,lamb=None, Fixed = False) 

CAG = CombActionGradient(actor, beta_mu, beta_std)

tot_accuracy = []
tot_actions = []

mean_rwd = 0

for ep in range(0,episodes):

    # Sample action from Gaussian policy
    action, mu_a, std_a = actor.computeAction(y_star, fixd_a_noise)

    # Perform action in the env
    true_y = model.step(action.detach())
    
    ## ====== Increase perturbation =======
    if ep > 2000 and ep % 40 == 1 and ep < 2300:
       perturbation += perturbation_increase
       print("\n Ep: ", ep, "Perturbation: ", perturbation, "\n")
    ## ==============================================

    # Add noise to sensory obs
    y = true_y + torch.randn_like(true_y) * sensory_noise + perturbation

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
    if ep > pre_train:
        est_y = estimated_model.step(action)  # re-estimate values since model has been updated
        CAG.update(y, est_y, action, mu_a, std_a, delta_rwd)

    if ep > 2300:
        accuracy = np.sqrt(rwd.detach().numpy())
        print("ep: ",ep)
        print("accuracy: ",accuracy)
        print("std_a: ", std_a,"\n")
        tot_accuracy.append(accuracy)
        tot_actions.append(action.detach().numpy())

print("Tot variability: ",np.std(tot_actions))
