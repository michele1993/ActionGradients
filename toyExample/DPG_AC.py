import torch
import torch.nn as nn
import torch.optim as opt

class Actor(nn.Module):

    def __init__(self,tot_a=1,input_s=1,out_put=1,ln_rate = 1e-3, trainable = True):

        super().__init__()

        self.tot_a = tot_a
        self.det_a = nn.Linear(input_s,out_put)

        #self.optimiser = opt.Adam(self.parameters(), ln_rate)

        if trainable:
            self.optimiser = opt.SGD(self.parameters(), ln_rate,momentum=0)
        else:

            for p in self.parameters():
                p.requires_grad = False

    def forward(self,y_star):

        return self.det_a(y_star)


    def update(self,loss):

        loss = torch.sum(loss)
        self.optimiser.zero_grad()
        loss.backward()
        ## ---------- Compute grad ---------------
        grad = self.det_a.bias.grad.detach().clone()
        ## --------------------------------------
        self.optimiser.step()

        return grad.item()



    def MB_update(self,gradient, action):

        #torch.autograd.backward(action, grad_tensors=gradient)
        action.backward(gradient= gradient)

        ## ---------- Compute grad ---------------
        grad = self.det_a.bias.grad.detach().clone()
        ## --------------------------------------
 
        self.optimiser.step()
        self.optimiser.zero_grad()

        return grad.item()


    def small_weight_init(self,l):

        if isinstance(l,nn.Linear):
            nn.init.normal_(l.weight,mean=0,std= 0.1)# std= 0.00005
            l.bias.data.fill_(0)

class Critic(nn.Module):

    def __init__(self, tot_a=1, input_s=1, out_put=1, ln_rate=1e-3):
        super().__init__()

        self.tot_a = tot_a
        #self.Q_est = nn.Linear(input_s, out_put)

        # Q has to be at least quadratic:
        #self.bias = nn.Parameter(torch.randn(1,))
        #self.linearP = nn.Parameter(torch.randn(1,))
        self.quadraticP = nn.Parameter(torch.randn(1, ))

        self.optimiser = opt.Adam(self.parameters(), ln_rate)

    def forward(self, action):

        return action**2 * self.quadraticP
        #return self.bias + action * self.linearP + action ** 2 * self.quadraticP
        #return self.Q_est(action)

    def small_weight_init(self, l):

        if isinstance(l, nn.Linear):
            nn.init.normal_(l.weight, mean=0, std=0.5)  # std= 0.00005
            nn.init.normal_(l.bias,mean=0, std=0.5)  # std= 0.00005

    def update(self, loss):

        loss = torch.sum(loss)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss
