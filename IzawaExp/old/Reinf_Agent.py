import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as opt

class R_agent(nn.Module):

    def __init__(self,std,tot_a=1,input_s=1,out_put=1,ln_rate = 1e-3):

        super().__init__()

        self.tot_a = tot_a
        self.std = std

        self.mu_s = nn.Linear(input_s,out_put)
        self.optimiser = opt.Adam(self.parameters(), ln_rate)

    def forward(self,y_star,test):

        x = self.mu_s(y_star)

        if not test:
            d = Normal(x, self.std)
            x = d.sample()
            self.log_ps = d.log_prob(x)

        return  x

    def update(self,rwd):

        loss = torch.sum(self.log_ps * rwd)
        self.optimiser.zero_grad()
        loss.backward()

        ## ========== Store gradients =========
        with torch.no_grad():
            grad_agent = self.mu_s.bias.grad.detach().clone()
        ## ===================================

        self.optimiser.step()
        self.log_ps = None # flush self.log just in case

        return loss, grad_agent.item()

    def small_weight_init(self,l):

        if isinstance(l,nn.Linear):
            nn.init.normal_(l.weight,mean=0,std= 0.1)# std= 0.00005
            l.bias.data.fill_(0)



