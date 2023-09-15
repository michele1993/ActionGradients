import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as opt
from torch.distributions import Categorical


class Actor(nn.Module):

    def __init__(self,state_s=2, action_s=2, hidden_size=56, output_size=2, ln_rate=1e-3, learn_std=True, max_angle=np.pi):

        super().__init__()
        self.num_layers=1 ## BAD!
        self.hidden_size = hidden_size
        self.max_angle = max_angle #NOTE: since output layer is tanh, max_angle = pi (i.e., can go from [-pi,pi])

        #self.embl = nn.Linear(lowest_n_moves, emb_s) # create an embedding layer to encode the planned seq of actions
        self.l1 = nn.LSTM(state_s, self.hidden_size, num_layers=self.num_layers, dropout=0, bidirectional=False)
        if learn_std:
            self.l2 = nn.Linear(hidden_size,output_size*2) # if learn standard deviation, also need to output it
        else:
            self.l2 = nn.Linear(hidden_size,output_size)

        self.optim = opt.Adam(self.parameters(),ln_rate) #, weight_decay=1e-3)

        self.log_ps = [] # Initialise list to store action prob within episode
        self.ill_log_ps = [] # Initialise list to store illegal action prob within episode

        # Initialise initial hidden state
        self.init_hidden()

    # Need to reset the hidden state at the end of each episode
    def init_hidden(self, batch_size=1):
        self.batch_size = batch_size
        self.hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
                            

    def forward(self,state):

        ## DELETE: ============
        #self.init_hidden() # transform RNN in a MLP by resetting hidden state at each step (unless pass entire seq in one step - not the case)
        ## ====================

        #NOTE: Output and hidden are the same in this case (where n_layer=1 and n_seq=1 for each forward pass)
        output,hidden,contex = self.l1(x,self.hidden_state,self.contex_state) 

        self.hidden_state = hidden # update hidden state
        self.contex_state = contex
        output = self.l2(output) 
        return torch.tanh(output) * self.max_angle

    def computeAction(self, x, fixd_a_noise):

        # Compute mean and log(std) of Gaussian policy
        mu_a, log_std_a = self(x) # Assume actor output log(std) so that can then transform it in positive value

        #Compute std
        std_a = torch.exp(log_std_a) # Need to initialise network to much smaller values

        #Entorpy bonus ?
        
        #Add fixd noise to exprl noise:
        action_std = std_a + fixd_a_noise 

        # Sample Gaussian perturbation
        a_noise = torch.randn(1) * action_std 

        # Compute action from sampled Gaussian policy
        return mu_a + a_noise, mu_a, action_std

    def ActionGrad_update(self,gradient, action):

        action.backward(gradient=gradient)
        #print("\n","Actor weight grad: ", self.l2.weight.grad)
        #print(,"Actor bias grad: ",self.l2.bias.grad)
        self.optimiser.step()
        self.optimiser.zero_grad()
