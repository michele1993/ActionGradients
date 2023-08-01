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
fixd_a_noise = 0.001
beta = 0


y_star = torch.zeros(1)

model = Mot_model()

agent = Actor(output_s=2, ln_rate = a_ln_rate, trainable = True)
estimated_model = Mot_model(ln_rate=model_ln_rate,lamb=None, Fixed = False) 

eps_rwd = []
tot_accuracy = []

for ep in range(1,episodes):

    # Compute mean of Gaussian policy
    mu_a, log_std_a = agent(y_star) # Assume actor output log(std) so that can then transform it in positive value
    std_a = torch.exp(log_std_a)
    
    #Add fixd noise to exprl noise:
    action_std = std_a + fixd_a_noise

    # Sample Gaussian perturbation
    a_noise = torch.randn(1) * action_std

    # Compute action from sampled Gaussian policy
    action = mu_a + a_noise

    # Perform action in the env
    true_y = model.step(action.detach())
    
    # Add noise to sensory obs
    y = true_y + torch.randn_like(true_y) * sensory_noise

    # Update the model
    est_y = estimated_model.step(action.detach())
    model_loss = estimated_model.update(y, est_y)

    # Compute differentiable rwd signal
    y.requires_grad_(True)
    rwd = (y - y_star)**2

    if ep > pre_train:

        ## ===== Compute MBDPG action gradient =========
        dr_dy = torch.autograd.grad(rwd, y)[0]
        est_y = estimated_model.step(action)  # re-estimate values since model has been updated
        # NOTE: here I diff relative to det_a instead of action, should be the same (since sigma is fixed)
        E_dr_dmu_a = torch.autograd.grad(est_y,mu_a,grad_outputs=dr_dy)[0] 

        #Error-based learning will try to converge to deterministic policy
        std_loss = action_std**2
        E_dr_dstd_a = torch.autograd.grad(std_loss, std_a)[0]

        #Combine two grads relative to mu and std into one vector
        E_grad = torch.stack([E_dr_dmu_a, E_dr_dstd_a])
        ## ============================================

        ## ===== Compute REINFORCE action gradient ========
        # Note: Here we are computing the gradients explicitly, so need to specify torch.no_grad()
        # it is not required above since use torch.autograd, which by default doesn't require_grad
        with torch.no_grad():
            # Mean action grad 
            R_dr_dmu_a = (-1/(2*action_std.detach()**2) * (action.detach() - mu_a)**2) * rwd.detach() 
            # Std action grad
            R_dr_dstd_a = rwd.detach() * (action_std**2 - (action.detach() - mu_a.detach())**2) / action_std**3

        #Combine two grads relative to mu and std into one vector
        R_grad = torch.cat([R_dr_dmu_a, R_dr_dstd_a])
        ## =========================================


        # Combine the two gradients
        comb_action_grad = beta * E_grad + (1-beta) * R_grad

        action_variables = torch.stack([mu_a, std_a])

        agent_grad = agent.ActionGrad_update(comb_action_grad, action_variables)

    eps_rwd.append(torch.sqrt(rwd))

    if ep % t_print == 0:

        print_acc = sum(eps_rwd) / t_print
        eps_rwd = []
        print("ep: ",ep)
        print("accuracy: ",print_acc,"\n")
        print("std_a: ", std_a,"\n")
        tot_accuracy.append(print_acc)
