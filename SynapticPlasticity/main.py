import os
from Linear_motor_model  import Mot_model
from Agent import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from CombinedAG import CombActionGradient
import matplotlib.pyplot as plt
import matplotlib as mpl

torch.manual_seed(0)

""" 
Show how sign of CB and sensory error control the sign of policy synaptic plasticity. To do so, assume there is a constant size sensory error
and manipulate sign of CB predictions to show how resulting action synaptic plasticity changes. 
I do it for both positive and negative directed sensory errors across two separate runs.
"""

trials = 100
change_model_ep = trials//2
t_print = 10
save_file = False
fix_sensory_errors = True

# Set noise variables
sensory_noise = 0.001
fixd_a_noise = 0.001 # set to experimental data value

# Set update variables
a_ln_rate = 0.1
c_ln_rate = 0.1
model_ln_rate = 0.01
beta = 1

## Peturbation:
target = 0.1056 # target angle : 6 degrees - Izawa and Shadmer, 2011
y_star = torch.tensor([target],dtype=torch.float32)

# Use two linear model to show CB may track different relations for different dynamics
# Linear model where y = a*w
model_1 = Mot_model()
# Linear model where y = -1*a*w
model_2 = Mot_model(reverse=True)

actor = Actor(ln_rate = a_ln_rate, trainable = True)

CAG = CombActionGradient(actor, beta)

mean_rwd = 0

synaptic_changes = []
dr_dy_s = []
dy_da_s = []

reverse_CB_eps = np.array([20,21,22,23,24,30,31,32,33,34,40,41,42,43,44])

model = model_1
for ep in range(0,trials):

    if ep == change_model_ep:
        model = model_2
        reverse_CB_eps = reverse_CB_eps + change_model_ep

    # Sample action from Gaussian policy
    action, mu_a, std_a = actor.computeAction(y_star, fixd_a_noise)

    # Perform action in the env
    true_y = model.step(action)
    
    # Add noise to sensory obs
    y = true_y + torch.randn_like(true_y) * sensory_noise 

    # Compute differentiable rwd signal
    rwd = (y - y_star)**2 # it is actually a punishment
    
    ## ====== Use running average to compute RPE =======
    delta_rwd = rwd - mean_rwd
    mean_rwd += c_ln_rate * delta_rwd.detach()
    ## ==============================================

    ## ====== Compute each gradient component individually ======

    # Sensory error (gradient)
    if fix_sensory_errors:
        dr_dy = 0.5
    else:
        dr_dy = torch.autograd.grad(delta_rwd, y, retain_graph=True)[0].squeeze().item() 

    # Sensitivity derivative (CB prediction)
    # Since we are in 1d can compute Jacobean explicitly in torch
    dy_da = torch.autograd.grad(y, action, retain_graph=True)[0] 

    if ep in reverse_CB_eps:
        dy_da *=-1
        
    # Compute overal synaptic gradient
    action.backward(gradient=dr_dy*dy_da) # compute the grad
    synaptic_grad = actor.l1.weight.grad # extract the grad
    actor.optimiser.zero_grad() # reset the grad

    # Append synaptic weight change
    dr_dy_s.append(dr_dy)
    dy_da_s.append(dy_da.squeeze().item())
    synaptic_changes.append(synaptic_grad.squeeze().item())
    ## =========================================

## ===== Plot the results ======
fig = plt.figure(figsize=(6, 4))
#gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[1,1])
gs = fig.add_gridspec(nrows=2, ncols=3, wspace=0.8, hspace=0.3, left=0.1, right=0.95, bottom=0.1, top=0.925)#, height_ratios=[1,0.2,1])

font_s = 7
mpl.rc('font', size=font_s)
plt.rcParams["font.family"] = "helvetica"
mpl.rcParams['xtick.labelsize'] = font_s 
mpl.rcParams['ytick.labelsize'] = font_s 

t = np.arange(0, change_model_ep,1)
data_1 = [dr_dy_s[:change_model_ep], dy_da_s[:change_model_ep], synaptic_changes[:change_model_ep]]
data_2 = [dr_dy_s[change_model_ep:], dy_da_s[change_model_ep:], synaptic_changes[change_model_ep:]]
titles = ['Cortical sensory error', 'CB prediction', 'Synaptic change\nat Dorsal Striatum'] 
y_label = ['Predicted\nerror', 'Predicted\nvalue', r'$ \Delta $ w']

## ---- 1st plot
#for data in [data_1,data_2]:
for e in range(len(data_1)):
#    for i in range(len(data)):
    i=0
    for data in [data_1,data_2]:
        ax = fig.add_subplot(gs[i,e])
        ax.plot(t, data[e], color='tab:orange', alpha=0.5, lw=1)
        ax.axhline(y=0, color='k', ls='--', lw=0.5)
        ax.set_ylabel(y_label[e])
        if e <2:
            ax.set_ylim([-1.5, 1.5])
        ax.set_xlim([0, change_model_ep-1])
        #ax.set_xticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if i ==0:
            ax.set_title(titles[e], fontsize=font_s) 

        ax.set_xlabel('Trials')
        i+=1



# Save image
if save_file:
    plt.savefig('/Users/px19783/Desktop/Synaptic_plasticity_1', format='png', dpi=1400)
else:
    plt.show()

## ===== Save results =========
# Create directory to store results
#file_dir = os.path.dirname(os.path.abspath(__file__))
#file_dir = os.path.join(file_dir,'results/model')
#
## Store model
#if beta ==0:
#    data = 'RBL_model.pt'
#elif beta ==1:    
#    data = 'EBL_model.pt'
#else:
#    data = 'Mixed_model.pt'
#model_dir = os.path.join(file_dir,data)
#
#if save_file:
#    # Create directory if it did't exist before
#    os.makedirs(file_dir, exist_ok=True)
#    torch.save({
#        'Actor': actor.state_dict(),
#        'Net_optim': actor.optimiser.state_dict(),
#        'Mean_rwd': mean_rwd,
#        'Est_model': estimated_model.state_dict(),
#        'Model_optim': estimated_model.optimiser.state_dict(),
#    }, model_dir)
