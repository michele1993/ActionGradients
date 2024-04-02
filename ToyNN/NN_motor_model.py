import torch
import numpy as np
import torch.nn as nn
import torch.optim as opt

class Mot_model(nn.Module):
    """ Implement a environment based on a random NN """

    def __init__(self, action_s, output_s, h_size = 256, ln_rate = 1e-3,  Fixed = True):

        super().__init__()

        self.l1 = nn.Linear(action_s,h_size)
        self.l2 = nn.Linear(h_size,output_s)

        if Fixed:
            print("fixed")
            for p in self.parameters():
                p_requires_grad_ = False
        else:
            self.optimiser = opt.Adam(self.parameters(), ln_rate)

        self.small_weight_init(self.parameters()) # Init std to neg value to get value below 1 when taking exp()

    def small_weight_init(self,l):
        if isinstance(l,nn.Linear):
            nn.init.normal_(l.weight,mean=0,std= 0.1)# std= 0.00005
            nn.init.normal_(l.bias,mean=0,std= 0.1)# std= 0.00005

    def step(self,action):
        x = self.l1(action)
        x = torch.relu(x)
        return torch.tanh(self.l2(x))

    def update(self, target, estimate):
        loss = torch.sum((target-estimate)**2)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss

