import os
from Linear_motor_model  import Mot_model
from Agent import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from CombinedAG import CombActionGradient
from torchmetrics.regression import CosineSimilarity
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

''' Train policies and forward models across different levels of sensory noises (and across different seeds) '''

seeds = [8721, 5467, 1092, 9372,2801]

trials = 10000
t_print = 1000
save_file = False

# Set noise variables
sensory_noises = torch.linspace(0.01,0.25,10)
noise = sensory_noises[-1]
fixd_a_noise = 0.02 # set to experimental data value

# Set update variables
a_ln_rate = 0.01
c_ln_rate = 0.1
model_ln_rate = 0.01
betas = np.arange(0,11,1) /10.0
rbl_weight = [1,1] #[1.5, 1.5]
ebl_weight = [1,1] #[0.1, 0.1]
#rbl_weight = [0.01, 0.01] #[1.5, 1.5]
#ebl_weight = [1,75] #[0.1, 0.1]

## Peturbation:
targets = [-30, -20, -10, 0, 10, 20, 30] # based on Izawa
y_star = torch.tensor(targets,dtype=torch.float32).unsqueeze(-1) * 0.0176

model = Mot_model()

cosine_sim = CosineSimilarity(reduction = 'mean') 

for s in seeds:
    torch.manual_seed(s)
    for b in betas:

        # Initialise differento components
        actor = Actor(action_s=1, ln_rate = a_ln_rate, trainable = True,opt_type='SGD') # 1D environment
        estimated_model = Mot_model(ln_rate=model_ln_rate,lamb=None, Fixed = False)
        CAG = CombActionGradient(actor, b, rbl_weight, ebl_weight)

        tot_accuracy = []
        tot_model_loss = []
        tot_grad_acc = []
        mean_rwd = 0
        trial_acc = []
        trial_model_loss = []
        trial_grad_acc = []

        for ep in range(1,trials+1):

            # Sample action from Gaussian policy
            action, mu_a, std_a = actor.computeAction(y_star, fixd_a_noise)

            # Perform action in the env
            true_y = model.step(action)
            true_rwd = (true_y - y_star)**2
            
            # Add noise to sensory obs
            y = true_y.detach() + torch.randn_like(true_y) * noise

            # Compute differentiable rwd signal
            y.requires_grad_(True)
            rwd = (y - y_star)**2 # it is actually a punishment
            trial_acc.append(torch.sqrt(true_rwd).detach().mean().item())
            
            ## ====== Use running average to compute RPE =======
            delta_rwd = rwd - mean_rwd
            mean_rwd += c_ln_rate * delta_rwd.detach()
            ## ==============================================

            # For rwd-base learning give rwd of 1 if reach better than previous else -1
            if b == 0:
               delta_rwd /= torch.abs(delta_rwd.detach()) 

            # Update the model
            est_y = estimated_model.step(action.detach())
            model_loss = estimated_model.update(y, est_y)
            trial_model_loss.append(model_loss.detach())

            # Update actor based on combined action gradient
            est_y = estimated_model.step(action)  # re-estimate values since model has been updated

            ## ======= Compute true gradient ======
            true_grad = CAG.computeEBLGrad(y=true_y, est_y=true_y, action=action, mu_a=mu_a, std_a=std_a, delta_rwd=true_rwd)
            ## ====================================

            est_grad = CAG.update(y, est_y, action, mu_a, std_a, delta_rwd)

            ## ====== Compare true gradient with estimated one =======
            #grad_sim = pairwise_cosine_similarity(est_grad,true_grad)#.mean()
            grad_sim = cosine_sim(est_grad,true_grad)#.mean()
            #grad_sim = ((est_grad - true_grad)**2).mean()
            trial_grad_acc.append(grad_sim.item())
            ## ====================================================

            # Store variables after pre-train (including final trials without a perturbation)
            if ep % t_print ==0:
                accuracy = sum(trial_acc) / len(trial_acc)
                model_loss = sum(trial_model_loss) / len(trial_model_loss)
                grad_acc = sum(trial_grad_acc) / len(trial_grad_acc)

                tot_accuracy.append(accuracy)
                tot_model_loss.append(model_loss)
                tot_grad_acc.append(grad_acc)
                trial_acc = []
                trial_model_loss = []
                trial_grad_acc = []

        print("Beta: ",b, "accuracy: ", accuracy, "Grad error: ", sum(tot_grad_acc)/len(tot_grad_acc))

        ## ===== Save results =========
        # Create directory to store results
        file_dir = os.path.dirname(os.path.abspath(__file__))
        file_dir = os.path.join(file_dir,'results','Noisy_Gradient',str(s))
        # Create directory if it did't exist before
        os.makedirs(file_dir, exist_ok=True)

        # Store model
        data = 'Beta_'+str(b)+'model.pt'
        model_dir = os.path.join(file_dir,data)

        if save_file:
            torch.save({
                "Targets": targets,
                'Actor': actor.state_dict(),
                'Net_optim': actor.optimiser.state_dict(),
                'Mean_rwd': mean_rwd,
                'Est_model': estimated_model.state_dict(),
                'Model_optim': estimated_model.optimiser.state_dict(),
            }, model_dir)

