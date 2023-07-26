import sys
sys.path.append('/Users/px19783/code_repository/cerebellum_project/ActionGradients')
from Motor_model  import Mot_model
from DPG_AC import *
import torch

 

torch.manual_seed(0)

episodes = 1000
action_noise = 0.1
sensory_noise = 0.05
a_ln_rate = 0.05
model_ln_rate = 0#.0005 # model_ln = 0.001
t_print = 10
pre_train = 0
sensory_noise = 0.05
action_std = 0.01
beta = 1


y_star = torch.zeros(1)

model = Mot_model()

agent = Actor(ln_rate = a_ln_rate, trainable = True)
estimated_model = Mot_model(ln_rate=model_ln_rate,lamb=None, Fixed = False) 

eps_rwd = []
tot_accuracy = []

for ep in range(1,episodes):

    # Compute mean of Gaussian policy
    det_a = agent(y_star)
    # Sample Gaussian perturbation
    a_noise = torch.randn(1) * action_std
    # Compute action from sampled Gaussian policy
    action = det_a + a_noise

    # Perform action in the env
    true_y = model.step(action.detach())
    
    # Add noise to sensory obs
    y = true_y + torch.randn_like(true_y) * sensory_noise

    # Update the model
    est_y = estimated_model.step(action.detach())
    model_loss = estimated_model.gradient_update(y, est_y)

    # Compute differentiable rwd signal
    y.requires_grad_(True)
    rwd = (y - y_star)**2

    if ep > pre_train:

        ## ===== Compute MBDPG action gradient =========
        dr_dy = torch.autograd.grad(rwd, y)[0]
        est_y = estimated_model.step(action)  # re-estimate values since model has been updated
        # NOTE: here I diff relative to det_a instead of action, should be the same (since sigma is fixed)
        dr_dy_da = torch.autograd.grad(est_y,det_a,grad_outputs=dr_dy)[0] 
        ## ============================================

        ## ===== Compute REINFORCE action gradient ========
        dr_da = (-1/(2*action_std**2) * (action.detach() - det_a)**2) * rwd.detach() 
        ## =========================================

        # Combine the two gradients
        comb_action_grad = beta * dr_dy_da + (1-beta) * dr_da 

        agent_grad = agent.MB_update(comb_action_grad, det_a)

    eps_rwd.append(torch.sqrt(rwd))

    if ep % t_print == 0:

        print_acc = sum(eps_rwd) / t_print
        eps_rwd = []
        print("ep: ",ep)
        print("accuracy: ",print_acc,"\n")
        tot_accuracy.append(print_acc)
