import os
from Motor_model  import Kinematic_model
from Forward_model import ForwardModel
from rnn_actor import Actor
import torch
import numpy as np
import matplotlib.pyplot as plt
from CombinedAG import CombActionGradient

""" Implement a line drawing task based on a 2D kinematic model - I follow the task of Boven et al., 2023 where the policy is a RNN that has to draw
    one of 6 possible traget straight lines only based on an inital cue. So, it is not a feedback model since the agent does not have access to the current state.
    However, I assume that the (learned) feedfoward model has access to the current state in order to compute the correct EBL gradients """
    



torch.manual_seed(0)

n_episodes = 2500
t_print = 100
save_file = False
action_s = 2 # two angles in 2D kinematic arm model
state_s = 2 # 2D space x,y-coord

# Set noise variables
sensory_noise = 0.01
fixd_a_noise = 0.02 # set to experimental data value

# Set update variables
a_ln_rate = 0.01
c_ln_rate = 0.1
model_ln_rate = 0.001
beta = 0.5
rbl_weight =  1.5
ebl_weight = 0.1

assert beta >= 0 and beta <= 1, "beta must be between 0 and 1 (inclusive)"

# Initialise env
model = Kinematic_model()

## ====== Generate 6 lists of targets (i.e. lines) ====== 
# All target lines start from the same initial point (x_0,y_0)
phi_0 = np.pi/2
# Compute min and max value within reaching space (this may exclude some reaching space, but doesn't matter)
small_circle_radius = model.l1 
max_coord = model.l1 + model.l2 

# Compute origin as point in front in the middle of reaching space
x_0 = 0
y_0 = (max_coord - small_circle_radius)/2 + small_circle_radius

# Generate n. target lines 
n_steps = 10 # based on Joe's paper
n_target_lines = 6
max_line_lenght = 0.2 # meters

assert max_line_lenght < np.abs(y_0 - small_circle_radius), "the max line lenght needs to be shorter or risk of going outside reacheable space"  

step_size = max_line_lenght / n_steps
radiuses = np.linspace(step_size, max_line_lenght+step_size, n_steps)
# Create n. equally spaced angle to create lines
ang_range = np.arange(0,n_target_lines) * (2 * np.pi) / n_target_lines

# Use unit circle to create lines by multiplying by growing radiuses to draw lines
cos_targ_xs = np.cos(ang_range)
sin_targ_ys = np.sin(ang_range)
x_targ = []
y_targ = []
# Use for loop to multiple each radius by corrspoding sine and cosine (should also be doable by proper np broadcasting)
for r in radiuses: 
    x_targ.append((cos_targ_xs * r)+x_0)
    y_targ.append((sin_targ_ys * r)+y_0)

x_targ = torch.tensor(np.array(x_targ)).T # shape:[batch, n_steps]
y_targ = torch.tensor(np.array(y_targ)).T # shape:[batch, n_steps]



## ======== Verification Plot ===========
# Check the targets are on 6 different lines
#plt.plot(x_targ,y_targ)
#plt.show()
## =============================

## ==== Initialise components ==========
estimated_model = ForwardModel(state_s=state_s,action_s=action_s, max_coord=max_coord, ln_rate=model_ln_rate)
actor = Actor(input_s= n_target_lines, batch_size=n_target_lines, ln_rate = a_ln_rate, learn_std=True)

CAG = CombActionGradient(actor, rbl_weight, ebl_weight)


tot_accuracy = []
mean_rwd = 0
trial_acc = []
model_losses = []
for ep in range(1,n_episodes+1):

    # Initialise cues at start of each trial
    cue = torch.eye(n_target_lines).unsqueeze(0) # each one-hot denotes different cue

    # Initialise starting position for each target line (start all from the same point)
    current_x = torch.tensor([x_0]).repeat(n_target_lines,1)
    current_y = torch.tensor([y_0]).repeat(n_target_lines,1)

    gradients = []
    action_variables []
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

        rwd = torch.sqrt((coord[:,0:1] - x_targ[:,t:t+1])**2 + (coord[:,1:2] - y_targ[:,t:t+1])**2) # it is actually a punishment

        trial_acc.append(torch.mean(rwd.detach()).item())
        
        ## ====== Use running average to compute RPE =======
        delta_rwd = rwd - mean_rwd
        mean_rwd += c_ln_rate * delta_rwd.detach()
        ## ==============================================

        # Update the model
        state_action = torch.cat([current_x,current_y,action],dim=1)
        est_coord = estimated_model.step(state_action.detach())
        model_loss = estimated_model.update(x_coord, y_coord, est_coord)
        model_losses.append(model_loss/n_steps)

        # Compute gradients and store them
        est_coord = estimated_model.step(state_action)  # re-estimate values since model has been updated
        R_grad = CAG.computeRBLGrad(action, mu_a, std_a, delta_rwd)
        E_grad = CAG.computeEBLGrad(y=coord, est_y=est_coord, action=action, mu_a=mu_a, std_a=std_a, delta_rwd=delta_rwd)

        gradients.append(beta * E_grad + (1-beta) * R_grad)
        action_variables.append(torch.cat([mu_a,std_a],dim=1))

        current_x = x_coord
        current_y = y_coord
        cue = torch.randn_like(n_target_lines).unsqueeze(0) # each one-hot denotes different cue

    CAG.update(gradients, action_variables)

    # Store variables after pre-train (including final trials without a perturbation)
    if ep % t_print ==0:
        accuracy = sum(trial_acc) / len(trial_acc)
        m_loss = sum(model_losses) / len(model_losses)
        print("ep: ",ep)
        #print("accuracy: ",accuracy)
        #print("std_a: ", std_a)
        print("Model Loss: ", m_loss,"\n")
        tot_accuracy.append(accuracy)
        trial_acc = []
        model_losses = []

## ===== Save results =========
# Create directory to store results
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'results/model')

# Store model
if beta ==0:
    data = 'RBL_model.pt'
elif beta ==1:    
    data = 'EBL_model.pt'
else:
    data = 'Mixed_model.pt'
model_dir = os.path.join(file_dir,data)

if save_file:
    # Create directory if it did't exist before
    os.makedirs(file_dir, exist_ok=True)
    torch.save({
        'Actor': actor.state_dict(),
        'Net_optim': actor.optimiser.state_dict(),
        'Mean_rwd': mean_rwd,
        'Est_model': estimated_model.state_dict(),
        'Model_optim': estimated_model.optimiser.state_dict(),
    }, model_dir)
