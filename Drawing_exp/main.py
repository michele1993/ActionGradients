import os
import sys
import argparse
from Kinematic_Motor_model  import Kinematic_model
from Forward_model import ForwardModel
from Gradient_model import GradientModel
from rnn_actor import Actor
import torch
import numpy as np
import matplotlib.pyplot as plt
from CombinedAG import CombActionGradient
from utils import compute_targetLines
import matplotlib as mpl

""" Implement a line drawing task based on a 2D kinematic model - I follow the task of Boven et al., 2023 where the policy is a RNN that has to draw
    one of 6 possible traget straight lines only based on an inital cue. So, it is not a feedback model since the agent does not have access to the current state.
    However, I assume that the (learned) feedfoward model has access to the current state in order to compute the correct EBL gradients """

parser = argparse.ArgumentParser()
parser.add_argument('--beta', '-b',type=float, nargs='?')
parser.add_argument('--step_x_update','-sxu', type=int, nargs='?', default=1)

seeds = [4634, 5637, 9920, 2310, 3021]

## Argparse variables:
args = parser.parse_args()
beta = args.beta
step_x_update = args.step_x_update

    
save_file = False
n_episodes = 10000  # 10000 NOTE: For beta=0.5 use n_episodes=5000 (ie.early stopping) 
model_pretrain = 100
grad_pretrain = model_pretrain * 1
t_print = 100
action_s = 2 # two angles in 2D kinematic arm model
state_s = 2 # 2D space x,y-coord

# Set noise variables
sensory_noise = 0.0001 #0.0001
fixd_a_noise = 0.0001 #.0002 # set to experimental data value

# Set update variables
assert beta >= 0 and beta <= 1, "beta must be between 0 and 1 (inclusive)"
actor_lr_decay = 1#0.99
gradModel_lr_decay = 1 #0.9
a_ln_rate = 0.001 #0.0025 
c_ln_rate = 0.1
model_ln_rate = 0.001
grad_model_ln_rate = 0.001

# Set experiment variables
n_target_lines = 6
n_steps = 10


# Initialise env
model = Kinematic_model()
large_circle_radium = model.l1 + model.l2 # maximum reacheable coordinate based on lenght of entire arm 
small_circle_radius = model.l1 # circle defined by radius of upper arm

## ====== Generate n. lists of targets (i.e. lines) ====== 

# All target lines start from the same initial origin point (x_0,y_0)
# Compute origin as point tanget to shoulder in the middle of reaching space (arbitary choice)
x_0 = 0
y_0 = (large_circle_radium - small_circle_radius)/2 + small_circle_radius
target_origin = [x_0,y_0]
line_lenght = 0.1 # length of each target line in meters

## ----- Check that target line length do not go outside reacheable space -----
distance = np.sqrt(x_0**2 + y_0**2) # assuming shoulder is at (0,0)
assert (line_lenght + distance) < large_circle_radium  and (distance-line_lenght) > small_circle_radius, "line_lenght needs to be shorter or risk of going outside reacheable space"  

x_targ, y_targ = compute_targetLines(target_origin, n_target_lines, n_steps, line_lenght) # shape: [n_target_lines, n_steps] , allowing batch training

## ==== Initialise components ==========

for s in seeds:
    torch.manual_seed(s)
    np.random.seed(s)

    ## Initialise different componets
    estimated_model = ForwardModel(state_s=state_s,action_s=action_s, max_coord=large_circle_radium, ln_rate=model_ln_rate)
    cerebellum = GradientModel(action_s=action_s, n_state_s=state_s, ln_rate=grad_model_ln_rate, lr_decay= gradModel_lr_decay)
    actor = Actor(input_s= n_target_lines, batch_size=n_target_lines, ln_rate = a_ln_rate, learn_std=True,lr_decay=actor_lr_decay)
    CAG = CombActionGradient(actor, action_s)

    tot_accuracy = []
    mean_rwd = 0
    trial_acc = []
    model_losses = []

    norm_ebl_gradients = []
    norm_rbl_gradients = []
    norm_EBL_tot_grad = []
    norm_RBL_tot_grad = []

    ebl_mu_gradients = []
    rbl_mu_gradients = []
    EBL_mu_tot_grad = []
    RBL_mu_tot_grad = []

    ## ====== Diagnostic variables ========
    #grad_model_loss = []
    #tot_grad_model_loss = []
    #target_ebl_grads = []
    #tot_target_ebl_grads = []
    ## ===================================

    ## TRIAL: Try passing the cue to the gradient NN (assume memory)
    #target_cue = torch.eye(n_target_lines).unsqueeze(0) # each one-hot denotes different cue

    ## ========== ADAM in action space =========
    #b_1 = 0.1 #0.9
    #b_2 = 0.999
    #M=0
    #R=0
    ## =========================================
    for ep in range(1,n_episodes+1):

        # Initialise cues at start of each trial
        cue = torch.eye(n_target_lines).unsqueeze(0) # each one-hot denotes different cue

        # Initialise starting position for each target line (start all from the same point)
        current_x = torch.tensor([x_0]).repeat(n_target_lines,1)
        current_y = torch.tensor([y_0]).repeat(n_target_lines,1)

        gradients = []
        action_variables = []

        # reset LSTM states for each trial
        actor.init_states()
        for t in range(n_steps):
            # Sample action from Gaussian policy
            action, mu_a, std_a = actor.computeAction(cue, fixd_a_noise)

            # Perform action in the env
            true_x_coord,true_y_coord = model.step(action.detach())

            # Add noise to sensory obs
            x_coord = true_x_coord + torch.randn_like(true_x_coord) * sensory_noise 
            y_coord = true_y_coord + torch.randn_like(true_y_coord) * sensory_noise 

            # Compute differentiable rwd signal
            coord = torch.cat([x_coord,y_coord], dim=1).requires_grad_(True)

            rwd = (coord[:,0:1] - x_targ[:,t:t+1])**2 + (coord[:,1:2] - y_targ[:,t:t+1])**2 # it is actually a punishment

            trial_acc.append(torch.mean(torch.sqrt(rwd.detach())).item())
            
            ## ====== Use running average to compute RPE =======
            delta_rwd = rwd.detach() - mean_rwd
            mean_rwd += c_ln_rate * delta_rwd.detach()
            ## ==============================================

            # Update the model
            state_action = torch.cat([current_x,current_y,action],dim=1)
            est_coord = estimated_model.step(state_action.detach())
            model_loss = estimated_model.update(x_coord.detach(), y_coord.detach(), est_coord)
            model_losses.append(model_loss.detach()/n_steps)

            if ep > model_pretrain and t % step_x_update == 0:
                # Compute gradients 
                est_coord = estimated_model.step(state_action)  # re-estimate values since model has been updated
                R_grad = CAG.computeRBLGrad(action, mu_a, std_a, delta_rwd)
                
                # Cortex-dependent gradient:
                dr_dy = CAG.compute_drdy(r=rwd,y=coord).unsqueeze(1)

                # ---- Cerebellum-dependent gradient: -----
                # Compute estimated dy_da by differentiating through forward model:
                dy_da = CAG.compute_dyda(y=est_coord, x=action)

                # Learn the dy/da
                est_dy_da = cerebellum(action.detach(),est_coord.detach()) # predict dy/da
                grad_loss = cerebellum.update(dy_da.view(n_target_lines,-1), est_dy_da)

                # Combine cerebellum and cortex gradients
                dr_dy_da = (dr_dy @ est_dy_da.view(dy_da.size()))

                ## ----- Interface to relate sampled actions to Gaussian policy variables (at the policy region) -----
                #NOTE: since dim(a) >1 the following two derivatives are Jacobians
                da_dmu = CAG.compute_da_dmu(action=action,mu=mu_a)
                da_dstd = CAG.compute_da_dstd(action=action,std=std_a)

                dr_dy_dmu = dr_dy_da @ da_dmu
                dr_dy_dstd = dr_dy_da @ da_dstd

                E_grad = torch.cat([dr_dy_dmu, dr_dy_dstd],dim=-1).squeeze().detach() # NOTE: I have checked that this gives the correct gradient

                # Store gradients
                if ep > grad_pretrain:

                    R_grad_norm = torch.norm(R_grad, dim=-1, keepdim=True)
                    E_grad_norm = torch.norm(E_grad, dim=-1, keepdim=True)

                    ## ======== Combine grad angle and norm separatedly ======
                    # Combine the two gradients angles
                    comb_action_grad = beta * E_grad/(E_grad_norm+1e-12) + (1-beta) * R_grad/(R_grad_norm +1e-12)
                    # Combine the two gradients norms
                    #comb_action_grad *= beta * E_grad_norm + (1-beta) * R_grad_norm
                    ## ======================================================

                    #comb_action_grad = beta * E_grad + (1-beta) * R_grad
                    comb_action_grad = beta * torch.clip(E_grad, -5,5) + (1-beta) * torch.clip(R_grad, -5, 5)
                    #comb_action_grad = beta * E_grad/(E_grad_norm+1e-12) + (1-beta) * R_grad/(R_grad_norm +1e-12)

                    ## ====== TRY: ADAM in action space: ======
                    #M = b_1 * M + (1-b_2) * comb_action_grad
                    #R = b_2 * R + (1-b_2) * comb_action_grad**2
                    #M_hat = M/(1 -(b_1)**action_s)
                    #R_hat = R/(1 -(b_2)**action_s)
                    #comb_action_grad = M_hat / torch.sqrt(R_hat+ 10e-8)
                    #comb_action_grad /= torch.sqrt(R_hat+ 10e-8) # WORK better
                    #M = torch.mean(M, dim=0, keepdim=True)
                    #R = torch.mean(R, dim=0, keepdim=True)
                    ## =======================================

                    #gradients.append(torch.clip(comb_action_grad,-5,5))
                    gradients.append(torch.clip(comb_action_grad,-5,5))
                    a_variab = torch.cat([mu_a,std_a],dim=1) 
                    action_variables.append(a_variab)
                    
                    # Store the gradient magnitude for plotting purposes
                    norm_ebl_gradients.append(E_grad_norm)
                    norm_rbl_gradients.append(R_grad_norm)

                    # Store the (vector) gradient of policy mean  for plotting purposes
                    ebl_mu_gradients.append(torch.mean(E_grad[:,0:2],dim=0))
                    rbl_mu_gradients.append(torch.mean(R_grad[:,0:2], dim=0))

            current_x = x_coord
            current_y = y_coord
            cue = torch.randn_like(cue) # each one-hot denotes different cue

        if gradients: 
            actor.ActionGrad_update(torch.cat(gradients), torch.cat(action_variables))


        # Store variables after pre-train (including final trials without a perturbation)
        if ep % t_print ==0:
            accuracy = sum(trial_acc) / len(trial_acc)
            m_loss = sum(model_losses) / len(model_losses)
            print("ep: ",ep)
            print("accuracy: ",accuracy)
            #print("std_a: ", std_a)
            #print("Model Loss: ", m_loss)
            tot_accuracy.append(accuracy)
            trial_acc = []
            model_losses = []

            ## Store gradients values for plotting purposes
            if norm_ebl_gradients:
                ## Update learning rate:
                #actor.scheduler.step()
                #cerebellum.scheduler.step()

                ## Store the norm of each action gradient
                norm_ebl_gradients = torch.cat(norm_ebl_gradients).mean()
                norm_rbl_gradients = torch.cat(norm_rbl_gradients).mean()
                norm_EBL_tot_grad.append(norm_ebl_gradients)
                norm_RBL_tot_grad.append(norm_rbl_gradients)
                norm_ebl_gradients = []
                norm_rbl_gradients = []

                ## Store the gradient relative to the action mean (i.e., a vector)
                ebl_mu_gradients = torch.stack(ebl_mu_gradients).mean(dim=0)
                rbl_mu_gradients = torch.stack(rbl_mu_gradients).mean(dim=0)
                EBL_mu_tot_grad.append(ebl_mu_gradients)
                RBL_mu_tot_grad.append(rbl_mu_gradients)
                ebl_mu_gradients = []
                rbl_mu_gradients = []
                
                ## ====== Diagnostic variables ========
                #target_eblGrad = torch.stack(target_ebl_grads).mean(dim=0).norm()
                #grad_loss = torch.cat(grad_model_loss).mean()
                ##print("\nTarget grad: ", target_eblGrad)
                #print("Grad Loss: ", grad_loss, "\n")
                #tot_target_ebl_grads.append(target_eblGrad)
                #tot_grad_model_loss.append(grad_loss) 
                #target_ebl_grads = []
                #grad_model_loss = []
                ## ===================================

    print(accuracy)

    ## ===== Save results =========
    # Create directory to store results
    run_dir = f'{step_x_update}_update'
    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_dir = os.path.join(file_dir,'results',run_dir,str(s))

    # Store model
    if beta ==0:
        data = 'RBL'
    elif beta ==1:    
        data = 'EBL'
    else:
        data = 'Mixed_'+str(beta)
    model_dir = os.path.join(file_dir,data+'_model.pt')

    current_x = torch.tensor([x_0]).repeat(n_target_lines,1)
    current_y = torch.tensor([y_0]).repeat(n_target_lines,1)
    origin = torch.cat([current_x,current_y],dim=1)
    if save_file:
        os.makedirs(file_dir, exist_ok=True)
        # Create directory if it did't exist before
        # Store command line
        command_line = f'Step x update: {step_x_update}, fixd_a_noise: {fixd_a_noise}, sensory_noise: {sensory_noise}, a_ln: {a_ln_rate}'
        with open(os.path.join(file_dir,'commands.txt'), 'w') as f:
            f.write(command_line)
        torch.save({
            'Accuracy': tot_accuracy,
            'Origin': origin,
            'Targets': torch.stack([x_targ,y_targ]),
            'Actor': actor.state_dict(),
            'Net_optim': actor.optimizer.state_dict(),
            'Est_model': estimated_model.state_dict(),
            'Model_optim': estimated_model.optimiser.state_dict(),
        }, model_dir)
        grads = torch.stack([torch.tensor(norm_RBL_tot_grad), torch.tensor(norm_EBL_tot_grad)])
        np.save(os.path.join(file_dir,data+'_gradients.npy'), grads.numpy())

## ===== Plot diagnostic data ======
#font_s =7
#mpl.rc('font', size=font_s)
#plt.rcParams["font.family"] = "helvetica"
#mpl.rcParams['xtick.labelsize'] = font_s 
#mpl.rcParams['ytick.labelsize'] = font_s 
#
#fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(7.5,3),
# gridspec_kw={'wspace': 0.35, 'hspace': 0.4, 'left': 0.07, 'right': 0.98, 'bottom': 0.15,
#                                               'top': 0.9})
#starting_ep = 0
#tot_accuracy = tot_accuracy[starting_ep:]
#t = torch.arange(len(tot_accuracy))
#axs[0].plot(t,tot_accuracy)
#axs[0].spines['right'].set_visible(False)
#axs[0].spines['top'].set_visible(False)
#axs[0].set_title("Accuracy",fontsize=font_s)
#axs[0].set_xticklabels([])
#
#tot_grad_model_loss = tot_grad_model_loss[starting_ep:]
#t = torch.arange(len(tot_grad_model_loss))
#axs[1].plot(t,tot_grad_model_loss)
#axs[1].spines['right'].set_visible(False)
#axs[1].spines['top'].set_visible(False)
#axs[1].set_title("l2 loss for gradient model",fontsize=font_s)
#axs[1].set_ylim([0.006,0.01])
#axs[1].set_xticklabels([])
#
#tot_target_ebl_grads = tot_target_ebl_grads[starting_ep:]
#t = torch.arange(len(tot_target_ebl_grads))
#axs[2].plot(t,tot_target_ebl_grads)
#axs[2].spines['right'].set_visible(False)
#axs[2].spines['top'].set_visible(False)
#axs[2].set_title("Target gradient magnitude",fontsize=font_s)
#axs[2].set_ylim([0.0001,0.01])
#axs[2].set_xticklabels([])
#plt.show()
# ==========================================
