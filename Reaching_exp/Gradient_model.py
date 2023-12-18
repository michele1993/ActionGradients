import torch
import numpy as np
import torch.nn as nn
import torch.optim as opt

class GradientModel(nn.Module):

    def __init__(self, action_s, n_state_s, ln_rate = 1e-3, h_state=56, lr_decay=0.9):

        super().__init__()

        self.l1 = nn.Linear(action_s+n_state_s,h_state) # takes the action and next state as input
        self.l2 = nn.Linear(h_state,action_s*n_state_s) # predicts Jacobian dy/da

        self.optimiser = opt.Adam(self.parameters(), ln_rate)#, weight_decay=1e-3)
        self.scheduler = opt.lr_scheduler.ExponentialLR(self.optimiser,gamma=lr_decay)

    def forward(self,action, n_state):

        x = torch.cat([action, n_state], dim=-1)
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
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
