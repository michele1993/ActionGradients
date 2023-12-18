import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt

class Actor(nn.Module):

    def __init__(self,input_s=1,action_s=2,ln_rate = 1e-3, hidden_size=56, learn_std=True, max_angle=np.pi,lr_decay=1):

        super().__init__()
        self.max_angle = max_angle
        self.learn_std = learn_std
        self.action_s = action_s

        self.l1 = nn.Linear(input_s,hidden_size)

        if learn_std:
            self.l2 = nn.Linear(hidden_size,action_s*2) # if learn standard deviation, also need to output it
        else:
            self.l2 = nn.Linear(hidden_size,action_s)

        self.optimizer = opt.Adam(self.parameters(),ln_rate) #, weight_decay=1e-3)
        #self.optimizer = opt.SGD(self.parameters(),ln_rate)
        self.scheduler = opt.lr_scheduler.ExponentialLR(self.optimizer,gamma=lr_decay)
        self.apply(self.small_weight_init)

    def small_weight_init(self,l):
        if isinstance(l,nn.Linear):
            nn.init.normal_(l.weight,mean=0,std= 0.1)# std= 0.00005
            nn.init.normal_(l.bias,mean=0,std= 0.1)# std= 0.00005

    def forward(self,x):

        x = self.l1(x)
        print(" CHECK ACTOR LINE 34, NO RELU USED !!!")
        exit()
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


class Critic(nn.Module):

    def __init__(self, tot_a=1, input_s=1, out_put=1, ln_rate=1e-3):
        super().__init__()

        self.tot_a = tot_a
        #self.Q_est = nn.Linear(input_s, out_put)

        # Q has to be at least quadratic:
        #self.bias = nn.Parameter(torch.randn(1,))
        #self.linearP = nn.Parameter(torch.randn(1,))
        self.quadraticP = nn.Parameter(torch.randn(1, )**2) # Initialise to positive value

        self.optimiser = opt.Adam(self.parameters(), ln_rate)

    def forward(self, action):

        return action**2 * self.quadraticP
        #return self.bias + action * self.linearP + action ** 2 * self.quadraticP
        #return self.Q_est(action)

    def small_weight_init(self, l):

        if isinstance(l, nn.Linear):
            nn.init.normal_(l.weight, mean=0, std=0.5)  # std= 0.00005
            nn.init.normal_(l.bias,mean=0, std=0.5)  # std= 0.00005

    def update(self, delta_rwd):

        loss = torch.sum(delta_rwd**2)
        self.optimiser.zero_grad()
        loss.backward(retain_graph=True) # Need this, since delta_rwd then used to update actor
        self.optimiser.step()

        return loss
