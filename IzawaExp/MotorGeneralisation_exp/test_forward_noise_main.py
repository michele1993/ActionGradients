import os
from Linear_motor_model  import Mot_model
from Agent import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from CombinedAG import CombActionGradient
from torchmetrics.regression import CosineSimilarity

''' Code to demonostrate correlation between Forward model accuracy and policy generalisation error. 
    Additionally, look at relation between train accuracy (based on training targets) and gradient accuracy
'''

torch.manual_seed(0)
np.random.seed(0)
seeds = [8721, 5467, 1092, 9372,2801]

save = False
# Set noise variables
sensory_noises = torch.linspace(0.01,0.25,10)
fixd_a_noise = 0.02 # set to experimental data value

# Set update variables
a_ln_rate = 0.01
model_ln_rate = 0.01
beta = 0.5
rbl_weight = [0.01, 0.01] #[1.5, 1.5]
ebl_weight = [1,75] #[0.1, 0.1]

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
test_seed_acc = []
test_seed_forward_acc = []
EBL_test_seed_gradSim = []
Mixed_test_seed_gradSim = []
RBL_test_seed_gradSim = []
# Training statistics (based on loaded models and training targets)
train_seed_acc = []
train_seed_forward_acc = []
train_seed_gradSim = []
for s in seeds:

    test_noise_acc = []
    test_forward_noise_acc = []
    EBL_test_noise_grad_sim = []
    Mixed_test_noise_grad_sim = []
    RBL_test_noise_grad_sim = []
    train_noise_acc = []
    train_forward_noise_acc = []
    train_noise_grad_sim = []

    for noise in sensory_noises:

        # Load models
        file_dir = os.path.dirname(os.path.abspath(__file__))
        file_dir = os.path.join(file_dir,'results','Noisy_Forward',str(s))
        data = 'Noise_'+str(round(noise.item(),3))+'model.pt'
        model_dir = os.path.join(file_dir,data)
        models = torch.load(model_dir)

        # Initialise Actor
        actor = Actor(action_s=1, ln_rate = a_ln_rate, trainable = True) # 1D environment
        actor.load_state_dict(models['Actor'])

        CAG = CombActionGradient(actor, beta, rbl_weight, ebl_weight)

        # Initialise FF model
        estimated_model = Mot_model(ln_rate=model_ln_rate,lamb=None, Fixed = False)
        estimated_model.load_state_dict(models['Est_model'])

        ## ======== Generalisation performance based on test targets ===============
        action, mu_a, std_a = actor.computeAction(test_y_star, fixd_a_noise)

        # Perform action in the env
        true_y = model.step(action)
        est_y = estimated_model.step(action)

        # Compute true forward error
        forward_error = (true_y - est_y)**2
        test_forward_noise_acc.append(forward_error.detach().mean())

        # Compute true rwd
        true_rwd = (true_y - test_y_star)**2 # it is actually a punishment
        test_noise_acc.append(torch.sqrt(true_rwd.detach()).mean().item())

        # Compute estimated gradient BASED ON NOISY sensory info
        y = true_y + torch.randn_like(true_y) * noise
        rwd = (y - test_y_star)**2 # it is actually a punishment

        delta_rwd = rwd # don't have access to expected rwd for new generalisation targets

        # Compute estimated gradient 
        EBL_grad = CAG.computeEBLGrad(y,est_y,action,mu_a, std_a,rwd) 
        # Compute true gradient 
        true_EBL_grad = CAG.computeEBLGrad(y=true_y, est_y=true_y, action=action, mu_a=mu_a, std_a=std_a, delta_rwd=true_rwd)

        # Compute RBL grad
        RBL_grad = CAG.computeRBLGrad(action, mu_a, std_a, delta_rwd)

        Mixed_grad = beta * EBL_grad + (1-beta) * RBL_grad
        #Mixed_grad = beta * EBL_grad/torch.norm(EBL_grad,dim=1,keepdim=True) + (1-beta) * RBL_grad/torch.norm(RBL_grad,dim=1,keepdim=True)


        # Compare true and estimated gradients
        EBL_grad_sim = cosine_sim(EBL_grad,true_EBL_grad) #Use cosine similarity since RBL and EBL 'cost' are different (i.e., RBL uses RPE)
        EBL_test_noise_grad_sim.append(EBL_grad_sim.item())

        Mixed_grad_sim = cosine_sim(Mixed_grad,true_EBL_grad) #Use cosine similarity since RBL and EBL 'cost' are different (i.e., RBL uses RPE)
        Mixed_test_noise_grad_sim.append(Mixed_grad_sim.item())

        RBL_grad_sim = cosine_sim(RBL_grad,true_EBL_grad) #Use cosine similarity since RBL and EBL 'cost' are different (i.e., RBL uses RPE)
        RBL_test_noise_grad_sim.append(RBL_grad_sim.item())
        ## =========================================================

        ## ========== Training performance with gradient ==================
        # base on loaded models and training targets
        action, mu_a, std_a = actor.computeAction(train_y_star, fixd_a_noise)

        # Perform action in the env
        true_y = model.step(action)
        est_y = estimated_model.step(action)

        # Compute true forward error
        forward_error = (true_y - est_y)**2
        train_forward_noise_acc.append(forward_error.detach().mean())

        # Compute true rwd
        true_rwd = (true_y - train_y_star)**2 # it is actually a punishment
        train_noise_acc.append(torch.sqrt(true_rwd.detach()).mean().item())

        # Compute estimated gradient BASED ON NOISY sensory info
        y = true_y + torch.randn_like(true_y) * noise
        rwd = (y - train_y_star)**2 # it is actually a punishment

        EBL_grad = CAG.computeEBLGrad(y,est_y,action,mu_a, std_a,rwd) 
        # Compute true gradient 
        true_EBL_grad = CAG.computeEBLGrad(y=true_y, est_y=true_y, action=action, mu_a=mu_a, std_a=std_a, delta_rwd=true_rwd)
        # Compare true and estimated gradients
        grad_sim = cosine_sim(EBL_grad,true_EBL_grad).mean() #Use cosine similarity since RBL and EBL 'cost' are different (i.e., RBL uses RPE)
        #grad_sim = ((EBL_grad-true_EBL_grad)**2).mean()

        train_noise_grad_sim.append(grad_sim.item())
        ## ============================================

    #print(noise)
    #print(test_noise_acc,"\n")

    # Store generalisation statistics
    test_seed_acc.append(np.array(test_noise_acc))
    test_seed_forward_acc.append(np.array(test_forward_noise_acc))
    EBL_test_seed_gradSim.append(np.array(EBL_test_noise_grad_sim))
    Mixed_test_seed_gradSim.append(np.array(Mixed_test_noise_grad_sim))
    RBL_test_seed_gradSim.append(np.array(RBL_test_noise_grad_sim))

    # Store training statistics
    train_seed_acc.append(np.array(train_noise_acc))
    train_seed_forward_acc.append(np.array(test_forward_noise_acc))
    train_seed_gradSim.append(np.array(train_noise_grad_sim))

## ====== GENERALISATION statistics =====
test_seed_acc = np.array(test_seed_acc)
test_seed_forward_acc = np.array(test_seed_forward_acc)
EBL_test_seed_gradSim = np.array(EBL_test_seed_gradSim)
Mixed_test_seed_gradSim = np.array(Mixed_test_seed_gradSim)
RBL_test_seed_gradSim = np.array(RBL_test_seed_gradSim)

## Select mean and std for corresponding values
test_mean_seed_acc = test_seed_acc.mean(axis=0)
test_std_seed_acc = test_seed_acc.std(axis=0)

test_mean_seed_forward_acc = test_seed_forward_acc.mean(axis=0)
test_std_seed_forward_acc = test_seed_forward_acc.std(axis=0)

EBL_test_mean_seed_gradSim = EBL_test_seed_gradSim.mean(axis=0)
EBL_test_std_seed_gradSim = EBL_test_seed_gradSim.std(axis=0)
Mixed_test_mean_seed_gradSim = Mixed_test_seed_gradSim.mean(axis=0)
Mixed_test_std_seed_gradSim = Mixed_test_seed_gradSim.std(axis=0)
RBL_test_mean_seed_gradSim = RBL_test_seed_gradSim.mean(axis=0)
RBL_test_std_seed_gradSim = RBL_test_seed_gradSim.std(axis=0)
## =======================================

#print('Test Mean error: ', test_mean_seed_acc,'\n')
#print('Test Mean forward model error: ', test_mean_seed_forward_acc,'\n')
#print('Test Mean EBL gradSim accuracy: ', EBL_test_mean_seed_gradSim,'\n')
#print('Test std EBL gradSim accuracy: ', EBL_test_std_seed_gradSim,'\n')
#print('Test Mean Mixed gradSim accuracy: ', Mixed_test_mean_seed_gradSim,'\n')
#print('Test std Mixed gradSim accuracy: ', Mixed_test_std_seed_gradSim/np.sqrt(5),'\n')
#print('Test Mean RBL gradSim accuracy: ', RBL_test_mean_seed_gradSim,'\n')


## ====== TRAINING statistics =====
train_seed_acc = np.array(train_seed_acc)
train_seed_forward_acc = np.array(train_seed_forward_acc)
train_seed_gradSim = np.array(train_seed_gradSim)

## Select mean and std for corresponding values
train_mean_seed_acc = train_seed_acc.mean(axis=0)
train_strain_seed_acc = train_seed_acc.std(axis=0)

train_mean_seed_forward_acc = train_seed_forward_acc.mean(axis=0)
train_std_seed_forward_acc = train_seed_forward_acc.std(axis=0)

train_mean_seed_gradSim = train_seed_gradSim.mean(axis=0)
train_strain_seed_gradSim = train_seed_gradSim.std(axis=0)
## =======================================

#print('\n\n')
#print('Train Mean error: ', train_mean_seed_acc,'\n')
#print('Train Mean forward: ', train_mean_seed_forward_acc)
#print('Train Mean gradSim: ', train_mean_seed_gradSim)

## ======= Save results ==============
test_data = np.stack((test_seed_acc, test_seed_forward_acc, test_seed_gradSim),axis=0)

file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'..','results','generalisation')
os.makedirs(file_dir, exist_ok=True)

## Also store data for forward model trained with noise but no noise at testing
outcome_dir = os.path.join(file_dir,'Noise_generalisation_statistics')

## ==== USE BELOW TO STORE DATA OF FORWARD MODEL TRAINED WITH NOISE BUT TESTED WITHOUT NOISE ====
#outcome_dir = os.path.join(file_dir,'Forward_Noise_Only_generalisation_statistics')
## =====================================

if save:
    np.save(outcome_dir,test_data)
