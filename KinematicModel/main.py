import os
from Motor_model  import Kinematic_model
from Forward_model import ForwardModel
from rnn_actor import Actor
import torch
import numpy as np
import matplotlib.pyplot as plt
from CombinedAG import CombActionGradient

torch.manual_seed(0)

n_episodes = 2500
t_print = 100
save_file = False
## Set trials to match Izawa and Shadmer, 2011 experimental set-up, where they add 1 degree pertubation every 40 trials up to 8 degreese

# Set noise variables
sensory_noise = 0.01
fixd_a_noise = 0.02 # set to experimental data value

# Set update variables
a_ln_rate = 0.01
c_ln_rate = 0.1
model_ln_rate = 0.01
beta_mu = 0.5
beta_std = beta_mu
rbl_std_weight =  1.5
ebl_std_weight = 0.1

## ==== Initialise components ==========
model = Kinematic_model()
estimated_model = ForwardModel()
actor = Actor( ln_rate = a_ln_rate, learn_std=True)

CAG = CombActionGradient(actor, beta_mu, beta_std, rbl_std_weight, ebl_std_weight)

## ====== Generate 6 lists of targets (i.e. lines) ====== 
# All target lines start from the same initial point (x_0,y_0)
phi_0 = np.pi/2
# Compute min and max value within reaching space (this may exclude some reaching space, but doesn't matter)
min_coord = model.l1 
max_coord = model.l1 + model.l2 

# Compute origin as point in front in the middle of reaching space
x_0 = 0
y_0 = (max_coord - min_coord)/2 + min_coord

# Generate n. target lines 
n_steps = 10 # based on Joe's paper
n_target_lines = 6
max_line_lenght = 0.2 # meters

assert max_line_lenght < np.abs(y_0 - min_coord), "the max line lenght needs to be shorter or risk of going outside reacheable space"  

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

x_targ = torch.tensor(np.array(x_targ))
y_targ = torch.tensor(np.array(y_targ))


## ======== Verification Plot ===========
# Check the targets are on 6 different lines
#plt.plot(x_targ,y_targ)
#plt.show()
## =============================

tot_accuracy = []
mean_rwd = 0
trial_acc = []
for ep in range(1,n_episodes+1):

    for t in range(n_steps):
        # Sample action from Gaussian policy
        action, mu_a, std_a = actor.computeAction(x, fixd_a_noise)

        # Perform action in the env
        true_y = model.step(action.detach())
        
        # Add noise to sensory obs
        y = true_y + torch.randn_like(true_y) * sensory_noise 

        # Compute differentiable rwd signal
        y.requires_grad_(True)
        rwd = (y - y_star)**2 # it is actually a punishment
        trial_acc.append(torch.sqrt(rwd.detach()).item())
        
        ## ====== Use running average to compute RPE =======
        delta_rwd = rwd - mean_rwd
        mean_rwd += c_ln_rate * delta_rwd.detach()
        ## ==============================================

        # Update the model
        est_y = estimated_model.step(action.detach())
        model_loss = estimated_model.update(y, est_y)

    # Update actor based on combined action gradient
    est_y = estimated_model.step(action)  # re-estimate values since model has been updated
    CAG.update(y, est_y, action, mu_a, std_a, delta_rwd)

    # Store variables after pre-train (including final trials without a perturbation)
    if ep % t_print ==0:
        accuracy = sum(trial_acc) / len(trial_acc)
        print("ep: ",ep)
        print("accuracy: ",accuracy)
        print("std_a: ", std_a,"\n")
        tot_accuracy.append(accuracy)
        trial_acc = []

## ===== Save results =========
# Create directory to store results
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'results/model')

# Store model
if beta_mu ==0:
    data = 'RBL_model.pt'
elif beta_mu ==1:    
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
