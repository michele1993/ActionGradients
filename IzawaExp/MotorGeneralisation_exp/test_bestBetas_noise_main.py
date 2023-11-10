import os
from Linear_motor_model  import Mot_model
from Agent import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from CombinedAG import CombActionGradient
from torchmetrics.regression import CosineSimilarity

''' Code to analyse whether the EBL gradient computation in the presence of fixed amount of sensory noise can be improved by introducing the RBL gradient
'''


seeds = [8721, 5467, 1092, 9372,2801]

save = True
# Set noise variables
sensory_noise = torch.linspace(0.001,0.1,100)
fixd_a_noise = 0.02 # set to experimental data value

# Set update variables
a_ln_rate = 0.01
model_ln_rate = 0.01
betas = torch.linspace(0,1,11)
rbl_weight = [1, 1] #[1.5, 1.5]
ebl_weight = [1,1] #[0.1, 0.1]

## Generate N test targets between 38 and -38
# since in Izawa's test based on pertub upto 8 degrees (and training was on max 30 degrees)
N = 100
max_val, min_val = 30,-30
range_size = (max_val - min_val)  # 2
test_targets = np.random.rand(N) * range_size + min_val
test_y_star = torch.tensor(test_targets,dtype=torch.float32).unsqueeze(-1) * 0.0176

# Define train targets
train_targets = [-30, -20, -10, 0, 10, 20, 30] # based on Izawa
train_y_star = torch.tensor(train_targets,dtype=torch.float32).unsqueeze(-1) * 0.0176

# Define cosine similarity function to be used to compare gradients
#cosine_sim = torch.nn.CosineSimilarity(dim=1)
cosine_sim = CosineSimilarity(reduction = 'mean') 

model = Mot_model()
# Generalisation statistics
seed_best_betas = []
for s in seeds:
    torch.manual_seed(s)
    np.random.seed(s)
    # Load Mixed policy trained with little noise 
    file_dir = os.path.dirname(os.path.abspath(__file__))
    #file_dir = os.path.join(file_dir,'results','Noisy_Forward',str(s))
    #data = 'Noise_'+str(round(noise.item(),3))+'model.pt'
    file_dir = os.path.join(file_dir,'results',str(s))
    data = 'Mixed_0.5model.pt'
    model_dir = os.path.join(file_dir,data)
    models = torch.load(model_dir)

    # Initialise Actor
    actor = Actor(action_s=1, ln_rate = a_ln_rate, trainable = True) # 1D environment
    actor.load_state_dict(models['Actor'])


    CAG = CombActionGradient(actor, 0, rbl_weight, ebl_weight)

    # Initialise FF model
    estimated_model = Mot_model(ln_rate=model_ln_rate,lamb=None, Fixed = False)
    estimated_model.load_state_dict(models['Est_model'])

    best_betas = []
    for noise in sensory_noise:
        
        ## ======== Generalisation performance based on test targets ===============
        action, mu_a, std_a = actor.computeAction(test_y_star, fixd_a_noise)

        # Perform action in the env
        true_y = model.step(action)
        est_y = estimated_model.step(action)

        # Compute true forward error
        forward_error = (true_y - est_y)**2

        # Compute true rwd
        true_rwd = (true_y - test_y_star)**2 # it is actually a punishment

        # Compute estimated gradient BASED ON NOISY sensory info
        y = true_y + torch.randn_like(true_y) * noise

        rwd = (y - test_y_star)**2 # it is actually a punishment

        delta_rwd = rwd # don't have access to expected rwd for new generalisation targets

        ## ==== Don't use binary feedback else different reward function ====
        # For rwd-base learning give rwd of 1 if reach better than previous else -1
        #if b == 0:
        #       delta_rwd /= torch.abs(delta_rwd.detach()) 
        ## ===================================

        # Compute estimated gradient 
        EBL_grad = CAG.computeEBLGrad(y,est_y,action,mu_a, std_a,delta_rwd) 
        #EBL_grad += torch.randn_like(EBL_grad) * noise

        # Compute true gradient 
        true_EBL_grad = CAG.computeEBLGrad(y=true_y, est_y=true_y, action=action, mu_a=mu_a, std_a=std_a, delta_rwd=true_rwd) 

        # Compute RBL grad
        RBL_grad = CAG.computeRBLGrad(action, mu_a, std_a, delta_rwd)

        # Compute Mixed grad for corresponding beta
        betas_grad_sim = []
        for b in betas: 
            # To understand the gradient interactions need to normalise them
            Mixed_grad = b * EBL_grad / torch.norm(true_EBL_grad,dim=1, keepdim=True) + (1-b) * RBL_grad / torch.norm(RBL_grad,dim=1,keepdim=True) 

            # Compare true and estimated gradients
            grad_sim = cosine_sim(Mixed_grad,true_EBL_grad)
            #grad_sim = torch.mean((true_EBL_grad[:,0] - Mixed_grad[:,0])**2)
            betas_grad_sim.append(grad_sim.item())
        best_betas.append(np.argmax(betas_grad_sim))
        #best_betas.append(np.array(betas_grad_sim))
        ## =========================================================


    # Store generalisation statistics
    seed_best_betas.append(np.array(best_betas))


seed_best_betas = np.array(seed_best_betas)
mean_best_beta = seed_best_betas.mean(axis=0)
std_best_beta = seed_best_betas.std(axis=0)
data = np.stack((sensory_noise,mean_best_beta, std_best_beta))
## ======= Save results ==============
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'..','results','generalisation')
os.makedirs(file_dir, exist_ok=True)
outcome_dir = os.path.join(file_dir,'Noise_generalisation_best_betas')
if save:
    np.save(outcome_dir,data)
