import torch
import numpy as np
import torch.nn as nn
import torch.optim as opt

class ForwardModel(nn.Module):

    def __init__(self,state_s, action_s, max_coord, ln_rate = 1e-3, h_state=56):

        super().__init__()

        self.max_coord = max_coord

        self.l1 = nn.Linear(state_s+action_s,h_state)
        self.l2 = nn.Linear(h_state,state_s)

        self.optimiser = opt.Adam(self.parameters(), ln_rate)

    def step(self,action):

        x = self.l1(action)
        x = torch.relu(x)
        x = self.l2(x) 
        x = torch.tanh(x) * self.max_coord
        return x

    def update(self, x_coord,y_coord, est_coord):

        est_x_coord, est_y_coord = est_coord[:,0:1], est_coord[:,1:2]
        loss = torch.mean((est_y_coord-y_coord)**2 + (est_x_coord-x_coord)**2)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss

    def weight_init(self, l):

        if isinstance(l, nn.Linear):
            l.weight.data.fill_(self.fixed_params[0]) #self.fixed_params[0]
            l.bias.data.fill_(self.fixed_params[1]) #self.fixed_params[1]
