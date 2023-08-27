import sys
sys.path.append('/Users/px19783/code_repository/cerebellum_project/ActionGradients')
from Motor_model  import Mot_model
from Agent import *
import torch
import numpy as np
from CombinedAG import CombActionGradient

 

torch.manual_seed(0)

episodes = 2000
a_ln_rate = 0.1
c_ln_rate = 0.1
model_ln_rate = 0.01
t_print = 100
pre_train = 200
sensory_noise = 0.001
fixd_a_noise = 0.03
beta_mu = 0
beta_std = 0


y_star = torch.ones(1)

model = Mot_model()

actor = Actor(output_s=2, ln_rate = a_ln_rate, trainable = True)
critic = Critic(ln_rate=c_ln_rate) # Initialise quadratic critic
estimated_model = Mot_model(ln_rate=model_ln_rate,lamb=None, Fixed = False) 

CAG = CombActionGradient(actor, beta_mu, beta_std)

ep_rwd = []
ep_actions = []
ep_critic_loss = []

tot_accuracy = []
tot_actions = []

mean_rwd = 0
critic_loss = 0

for ep in range(0,episodes):

    # Sample action from Gaussian policy
    action, mu_a, std_a = actor.computeAction(y_star, fixd_a_noise)
    ep_actions.append(action.detach().numpy())

    # Perform action in the env
    true_y = model.step(action.detach())
    
    # Add noise to sensory obs
    y = true_y + torch.randn_like(true_y) * sensory_noise

    # Compute differentiable rwd signal
    y.requires_grad_(True)
    rwd = (y - y_star)**2 # it is actually a punishment
    
    ## ====== Use running average to compute RPE =======
    delta_rwd = rwd - mean_rwd
    mean_rwd += c_ln_rate * delta_rwd.detach()
    ## ==============================================

    ## ====== Use critic to compute RPE =======
    #pred_rwd = critic(action)
    #delta_rwd = rwd - pred_rwd
    # Update critic
    #critic_loss = critic.update(delta_rwd)
    ## ==============================================


    # Update the model
    est_y = estimated_model.step(action.detach())
    model_loss = estimated_model.update(y, est_y)

    # Update actor based on combined action gradient
    if ep > pre_train:
        est_y = estimated_model.step(action)  # re-estimate values since model has been updated
        CAG.update(y, est_y, action, mu_a, std_a, delta_rwd)

    ep_rwd.append(torch.sqrt(rwd))
    ep_critic_loss.append(critic_loss)

    if ep % t_print == 0:

        print_acc = sum(ep_rwd) / len(ep_rwd)
        action_var = np.var(ep_actions)
        print_critic_loss = sum(ep_critic_loss) / len(ep_critic_loss)
        ep_rwd = []
        ep_actions = []
        ep_critic_loss = []
        print("ep: ",ep)
        print("accuracy: ",print_acc)
        print("critic loss: ", print_critic_loss)
        print("std_a: ", std_a,"\n")
        tot_accuracy.append(print_acc)
        tot_actions.append(action_var)

print("Tot variance: ",np.mean(tot_actions))
