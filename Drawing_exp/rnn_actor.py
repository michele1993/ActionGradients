import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as opt
from torch.distributions import Categorical


class Actor(nn.Module):

    def __init__(self,input_s=6, action_s=2, batch_size=1,hidden_size=56, ln_rate=1e-3, learn_std=True, max_angle=np.pi, lr_decay=0.97):

        super().__init__()

        self.action_s = action_s
        self.num_layers=1 ## BAD!
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learn_std = learn_std
        self.max_angle = max_angle #NOTE: since output layer is tanh, max_angle = pi (i.e., can go from [-pi,pi])

        self.l1 = nn.LSTM(input_s, self.hidden_size, num_layers=self.num_layers, dropout=0, bidirectional=False)

        if learn_std:
            self.l2 = nn.Linear(hidden_size,action_s*2) # if learn standard deviation, also need to output it
        else:
            self.l2 = nn.Linear(hidden_size,action_s)

        self.optimizer = opt.Adam(self.parameters(),ln_rate) #, weight_decay=1e-3)
        self.scheduler = opt.lr_scheduler.ExponentialLR(self.optimizer,gamma=lr_decay)

        # Initialise initial hidden state
        self.init_states()

    # Need to reset the hidden state at the end of each episode
    def init_states(self):
        self.hidden_state = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        self.contex_state = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

    def forward(self,x):

        ## DELETE: ============
        #self.init_hidden() # transform RNN in a MLP by resetting hidden state at each step (unless pass entire seq in one step - not the case)
        ## ====================

        #NOTE: Output and hidden are the same in this case (where n_layer=1 and n_seq=1 for each forward pass)
        output,(h,c)= self.l1(x,(self.hidden_state,self.contex_state))
        self.hidden_state = h # update hidden state
        self.contex_state = c

        output = self.l2(output) 

        if self.learn_std:
            mu_a = output[...,:self.action_s].squeeze()
            log_std_a = output[...,self.action_s:].squeeze()
        else:
            mu_a = output.squeeze()
            log_std_a = torch.zeros_like(mu_a).squeeze()

        return torch.tanh(mu_a) * self.max_angle, log_std_a

    def computeAction(self, x, fixd_a_noise):

        # Compute mean and log(std) of Gaussian policy
        mu_a, log_std_a = self(x) # Assume actor output log(std) so that can then transform it in positive value

        #Compute std
        std_a = torch.exp(log_std_a) # Need to initialise network to much smaller values

        #Entorpy bonus ?
        
        #Add fixd noise to exprl noise:
        action_std = std_a + fixd_a_noise 

        # Sample Gaussian perturbation
        a_noise = torch.randn_like(mu_a) * action_std 

        # Compute action from sampled Gaussian policy
        return torch.clip(mu_a + a_noise, min=-self.max_angle, max=self.max_angle), mu_a, action_std

    def ActionGrad_update(self,gradient, action):

        action.backward(gradient=gradient)
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Refresh h state after each update
        self.init_states()
