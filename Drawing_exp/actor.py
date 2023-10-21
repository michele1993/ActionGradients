import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt

class Actor(nn.Module):

    def __init__(self,input_s=1,action_s=2,ln_rate = 1e-3, hidden_size=56, learn_std=True, max_angle=np.pi):

        super().__init__()
        self.max_angle = max_angle
        self.learn_std = learn_std
        self.action_s = action_s

        self.l1 = nn.Linear(input_s,hidden_size)

        if learn_std:
            self.l2 = nn.Linear(hidden_size,action_s*2) # if learn standard deviation, also need to output it
        else:
            self.l2 = nn.Linear(hidden_size,action_s)

        #self.optimizer = opt.Adam(self.parameters(),ln_rate) #, weight_decay=1e-3)
        self.optimizer = opt.SGD(self.parameters(),ln_rate)

    def forward(self,x):

        x = self.l1(x)
        output = self.l2(x)

        if self.learn_std:
            mu_a = output[...,:self.action_s]
            log_std_a = output[...,self.action_s:]
        else:
            mu_a = output
            log_std_a = torch.zeros_like(mu_a)

        return torch.tanh(mu_a) * self.max_angle, log_std_a

    
    def computeAction(self, y_star, fixd_a_noise):

        # Compute mean and log(std) of Gaussian policy
        mu_a, log_std_a = self(y_star) # Assume actor output log(std) so that can then transform it in positive value

        #Compute std
        std_a = torch.exp(log_std_a) # Need to initialise network to much smaller values
        
        #Add fixd noise to exprl noise:
        action_std = std_a + fixd_a_noise 

        # Sample Gaussian perturbation
        a_noise = torch.randn_like(mu_a) * action_std 

        # Compute action from sampled Gaussian policy
        return torch.clip(mu_a + a_noise, min=-self.max_angle, max=self.max_angle), mu_a, action_std

    def update(self,loss):

        loss = torch.sum(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def ActionGrad_update(self,gradients, action):

        action.backward(gradient=gradients)
        self.optimizer.step()
        self.optimizer.zero_grad()


    def small_weight_init(self,l):

        if isinstance(l,nn.Linear):
            #nn.init.normal_(l.weight,mean=0,std= 0.001)# std= 0.00005
            #l.bias.data.fill_(-2.5) # initialise to neg value so exp() return value < 1
            l.bias.data.fill_(-1) # initialise to neg value so exp() return value < 1
            l.weight.data.fill_(-1) # initialise to neg value so exp() return value < 1

class Critic(nn.Module):

    def __init__(self, input_s=1, output_s=1, h_state=116, ln_rate=1e-3):
        super().__init__()

        self.l1 = nn.Linear(input_s,h_state)
        self.l2 = nn.Linear(h_state,output_s) 

        self.optimiser = opt.Adam(self.parameters(), ln_rate)

    def forward(self, x):

        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)

        return x

    def small_weight_init(self, l):

        if isinstance(l, nn.Linear):
            nn.init.normal_(l.weight, mean=0, std=0.5)  # std= 0.00005
            nn.init.normal_(l.bias,mean=0, std=0.5)  # std= 0.00005

    def update(self, target, estimate):

        loss = torch.sum((target - estimate)**2)
        self.optimiser.zero_grad()
        loss.backward(retain_graph=True) # Need this, since delta_rwd then used to update actor
        self.optimiser.step()

        return loss
