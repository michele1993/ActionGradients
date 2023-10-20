import torch
import numpy as np
import torch.nn as nn
import torch.optim as opt

class GradientModel(nn.Module):

    def __init__(self,state_s, action_s, ln_rate = 1e-3, h_state=56, target_size=2, lr_decay=0.9):

        super().__init__()


        self.l1 = nn.Linear(state_s+action_s+target_size,h_state) # takes the state, action and reward as input
        self.l2 = nn.Linear(h_state,action_s*2) # need to multiply by 2 since have mean and std of Gaussian policy

        self.optimiser = opt.Adam(self.parameters(), ln_rate)#, weight_decay=1e-3)
        self.scheduler = opt.lr_scheduler.ExponentialLR(self.optimiser,gamma=lr_decay)

    def forward(self,state_action, target):

        x = torch.cat([state_action,target], dim=-1)
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        #x = torch.tanh(x) #* 0.5
        return x

    def update(self, target_grad, est_grad):

        loss = torch.mean((target_grad - est_grad)**2)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss

    def weight_init(self, l):

        if isinstance(l, nn.Linear):
            l.weight.data.fill_(self.fixed_params[0]) #self.fixed_params[0]
            l.bias.data.fill_(self.fixed_params[1]) #self.fixed_params[1]
