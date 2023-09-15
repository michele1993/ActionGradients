import torch
import numpy as np
import torch.nn as nn
import torch.optim as opt

class ForwardModel(nn.Module):

    def __init__(self,ln_rate = 1e-3, action_s =2, output_s=2, h_state=56):

        super().__init__()

        self.l1 = nn.Linear(action_s,h_state)
        self.l2 = nn.Linear(h_state,output_s)

        self.optimiser = opt.Adam(self.parameters(), ln_rate)

    def step(self,action):

        x = self.l1(action)
        x = torch.relu(x)
        x,y = self.l2(x) 

        return x,y

    def update(self, target, estimate):

        loss = torch.sum((target-estimate)**2)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss

    def weight_init(self, l):

        if isinstance(l, nn.Linear):
            l.weight.data.fill_(self.fixed_params[0]) #self.fixed_params[0]
            l.bias.data.fill_(self.fixed_params[1]) #self.fixed_params[1]
