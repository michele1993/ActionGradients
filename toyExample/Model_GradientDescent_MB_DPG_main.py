import sys
sys.path.append('/Users/px19783/code_repository/Simple_motor_model')

from Motor_model import Mot_model
from DPG.DPG_AC import *
import torch

# Better version, used in the plots

# Test MB_DPG on mirror-reversal, but learning the model through gradient descent, rather than analytically,
# This is much better, since no need to consider past 1000 episodes with decaying value as well as reflecting
# the same implementation used for the Two-joint arm model; a_ln_rate = 0.01; model_ln_rate = 0.0005 (same as supervised)


torch.manual_seed(0)

# works best with below parameters, with noise on the outcome
episodes = 1500
a_ln_rate = 0.01# best: 0.01#0.005 # 0.01
model_ln_rate = 0.0005 # model_ln = 0.001
t_print = 10
buffer_size = 1
pre_train = 0#20000
sensory_noise = 0.05#.025
action_noise = 0#.025
reverse_acc = 0.1

y_star = torch.zeros(1)

model = Mot_model()

# Trainable components
agent = Actor(ln_rate = a_ln_rate, trainable = True)# False, use False if update actor manually, without opt

estimated_model = Mot_model(ln_rate=model_ln_rate,lamb=None, Fixed = False) #lamb=0.9999


true_eps_rwd = []

tot_accuracy = []
ep_actions = []
tot_actions = []
tot_grads = []
tot_model_loss = []

ep_model_loss = []
reverse = True
m_update = True
revers_activated = False             

print_True_acc = 10
actions = []
grads = []

switch_counter = 0


# Store the position of the arm and the mirror position:
y_pos = []
tot_y_pos = []
y_hat_pos = []
tot_y_hat_pos = []
true_y_pos = []
true_tot_y_pos = []

for ep in range(1,episodes):

    det_a = agent(y_star)

    if ep > pre_train:
        action = det_a
        t_print = 10
    else:
        action = det_a + torch.randn(1) * 0.1

    a_noise = torch.randn(1) * action_noise

    ep_actions.append(action.detach() +a_noise)


    true_y = model.step(action.detach() + a_noise)

    # Store the true arm position:
    if revers_activated: 
        true_y_pos.append(true_y.detach().item() * (-1))
    else:
        true_y_pos.append(true_y.detach().item())

    # add noise to the outcome
    y = true_y + torch.randn_like(true_y) * sensory_noise

    actions.append(det_a.detach())
    y_pos.append(y.detach().item())

    # estimate gradient through autograd function
    est_y = estimated_model.step(action.detach())

    y_hat_pos.append(est_y.detach().item())

    model_loss = estimated_model.gradient_update(y, est_y)
    ep_model_loss.append(model_loss)

    y.requires_grad_(True)

    rwd = (y_star - y) ** 2

    true_eps_rwd.append(torch.sqrt(rwd).detach())

    if ep > pre_train:

        dr_dy = torch.autograd.grad(rwd, y)[0]

        est_y = estimated_model.step(action)  # re-estimate values since model has been updated

        dyh_da = torch.autograd.grad(est_y,action,grad_outputs=dr_dy)[0] # multiple dr_dy by dyh_da

        agent_grad = agent.MB_update(dyh_da,action)
        grads.append(agent_grad)



    if  ep % t_print == 0:

        print_True_acc = sum(true_eps_rwd) / t_print
        avr_action = sum(ep_actions) / t_print
        avr_model_loss = sum(ep_model_loss)/t_print

        if reverse:
            switch_counter +=1

        print("ep: ",ep)
        print("True accuracy: ",print_True_acc,)
        print("Model loss: ", avr_model_loss,"\n")
        true_eps_rwd = []
        ep_model_loss = []
        ep_actions = []
        tot_accuracy.append(print_True_acc)
        tot_actions.append(avr_action)
        tot_grads.append(sum(grads)/t_print)
        tot_model_loss.append(avr_model_loss)
        tot_y_pos.append(sum(y_pos)/t_print)
        tot_y_hat_pos.append(sum(y_hat_pos)/t_print)
        true_tot_y_pos.append(sum(true_y_pos)/t_print)
        grads = []
        y_pos = [] 
        y_hat_pos = []
        true_y_pos = []



    # use ep > 1500 to show longer convergence beofre the mirrow revelrsal
    if print_True_acc < reverse_acc and reverse: #ep > 1500 and reverse:#print_acc < 0.1 and reverse: # print_acc < 0.005
        reverse = False
        revers_activated = True             
        print("REVERSED","\n")
        model.mirror_reversal()
        switch_ep=ep
        #m_update = False # Uncomment to block model learning after reversal



torch.save(tot_grads, "/Users/px19783/code_repository/Simple_motor_model/Results/MBDPG/MBDPG_grads_s0_ModelGD")
torch.save(tot_y_pos, "/Users/px19783/code_repository/Simple_motor_model/Results/MBDPG/MBDPG_y_s0_ModelGD")
torch.save(true_tot_y_pos, "/Users/px19783/code_repository/Simple_motor_model/Results/MBDPG/MBDPG_true_y_s0_ModelGD")
torch.save(tot_y_hat_pos, "/Users/px19783/code_repository/Simple_motor_model/Results/MBDPG/MBDPG_yHat_s0_ModelGD")

torch.save(tot_accuracy, "/Users/px19783/code_repository/Simple_motor_model/Results/MBDPG/MBDPG_accuracy_new_s0_ModelGD")
torch.save(tot_actions, "/Users/px19783/code_repository/Simple_motor_model/Results/MBDPG/MBDPG_actions_new_s0_ModelGD")
torch.save(switch_ep//t_print, "/Users/px19783/code_repository/Simple_motor_model/Results/MBDPG/MBDPG_epSwitch_s0_ModelGD")
