import torch
import numpy as np
import torch.nn as nn
import torch.optim as opt

class Mot_model(nn.Module):

    def __init__(self,ln_rate = 1e-3,lamb=1, action_s =1, output_s=1, Fixed = True, fixed_params = [1,0]):

        super().__init__()

        self.lamb = lamb
        self.model = nn.Linear(action_s,output_s)
        self.fixed_params = fixed_params

        if Fixed:
            self.apply(self.F_weight_init)
            #print("fixed")
        else:
            self.apply(self.weight_init)
            #self.optimiser = opt.SGD(self.parameters(), ln_rate, momentum=0)
            self.optimiser = opt.Adam(self.parameters(), ln_rate)

    def step(self,action):

        return self.model(action)

    def update(self, target, estimate):

        loss = torch.sum((target-estimate)**2)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss



    def F_weight_init(self, l):

        if isinstance(l, nn.Linear):
            l.weight.data.fill_(self.fixed_params[0])
            l.weight.requires_grad = False
            l.bias.data.fill_(self.fixed_params[1])
            l.bias.requires_grad = False

    def weight_init(self, l):

        if isinstance(l, nn.Linear):
            l.weight.data.fill_(self.fixed_params[0]) #self.fixed_params[0]
            l.bias.data.fill_(self.fixed_params[1]) #self.fixed_params[1]


    # This methods is for the model-based
    def analytic_update(self, states,actions, n_states):

        #n_states = len(states)

        states = states[-n_states:]
        actions = actions[-n_states:]
        # Compute diagonal matrix with time-decaying weights
        expnts = np.linspace(n_states-1,0,n_states)
        w_matrix = np.zeros((n_states,n_states))
        weights = self.lamb ** expnts
        np.fill_diagonal(w_matrix,weights)

        X = np.stack([np.array(actions),np.ones(n_states)]) # Since parameters()contain slope first and then bias
        Y = np.array(states).reshape(-1,1)

        XW = np.matmul(X,w_matrix)

        XWX = np.matmul(XW,X.T)
        XWY = np.matmul(XW,Y)

        inv_XWX = np.linalg.inv(XWX)
        opt_w = np.matmul(inv_XWX,XWY)

        self.model.weight.data.fill_(opt_w[0,0])
        self.model.bias.data.fill_(opt_w[1,0])

        # i =0
        # for p in self.parameters():
        #     p.data = torch.from_numpy(opt_w[i]).float()
        #     i +=1


    def compute_optimal_a(self,y_star):

        slope, bias = self.parameters()

        return y_star/slope - bias/slope






