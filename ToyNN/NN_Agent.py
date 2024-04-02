import torch
import torch.nn as nn
import torch.optim as opt

class Actor(nn.Module):

    def __init__(self,input_s=1,action_s=1,h_size =56, ln_rate = 1e-3):

        super().__init__()

        self.action_s = action_s

        self.l1 = nn.Linear(input_s, h_size)
        self.l2 = nn.Linear(h_size, action_s*2) # don't lear std of gaussian policy

        # Initialise network with small weights
        self.small_weight_init(self.parameters()) # Init std to neg value to get value below 1 when taking exp()

        self.optimiser = opt.Adam(self.parameters(), ln_rate)

    def forward(self,y_star):
        x = self.l1(y_star)
        x = torch.relu(x)
        x = self.l2(x)

        mu_a = x[...,:self.action_s]
        log_std_a = x[...,self.action_s:]

        return mu_a, log_std_a

    def computeAction(self, y_star, fixd_a_noise):

        # Compute mean 
        mu_a, log_std_a = self(y_star) # Assume actor output log(std) so that can then transform it in positive value

        #Add fixd noise to exprl noise:
        action_std = torch.exp(log_std_a) + fixd_a_noise

        # Sample Gaussian perturbation
        a_noise = torch.randn_like(mu_a) * action_std 

        # Compute action from sampled Gaussian policy
        return mu_a + a_noise, mu_a, action_std
   
    def ActionGrad_update(self,gradient, action):
        action.backward(gradient=gradient)
        self.optimiser.step()
        self.optimiser.zero_grad()

    def small_weight_init(self,l):
        if isinstance(l,nn.Linear):
            nn.init.normal_(l.weight,mean=0,std= 0.1)# std= 0.00005
            nn.init.normal_(l.bias,mean=0,std= 0.1)# std= 0.00005
