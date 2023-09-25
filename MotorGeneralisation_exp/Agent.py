import torch
import torch.nn as nn
import torch.optim as opt

class Actor(nn.Module):

    def __init__(self,input_s=1,action_s=1,ln_rate = 1e-3, trainable = True, opt_type='Adam'):

        super().__init__()

        self.action_s = action_s
        # Maintain two separate sets of weights for the mean and std of the Gaussian policy
        # So can initialise them with two different scales
        self.l1 = nn.Linear(input_s,action_s)
        self.l2 = nn.Linear(input_s, action_s)


        # Initialise network with small weights
        #nn.init.normal_(self.l2.weight, mean=0, std=0.01)  # Initialize with a small scale
        #nn.init.normal_(self.l2.bias, mean=0, std=0.01)  # Initialize with a small scale
        self.small_weight_init(self.l2) # Init std to neg value to get value below 1 when taking exp()

        if trainable:
            if opt_type=="SGD":
                self.optimiser = opt.SGD(self.parameters(), ln_rate)#, momentum=0.9)
            else:
                self.optimiser = opt.Adam(self.parameters(), ln_rate)
        else:

            for p in self.parameters():
                p.requires_grad = False

    def forward(self,y_star):

        x_1 = self.l1(y_star)
        x_2 = self.l2(y_star)
        return x_1,x_2
    
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
        return mu_a + a_noise, mu_a, action_std

    def update(self,loss):

        loss = torch.sum(loss)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


    def ActionGrad_update(self,gradient, action):

        action.backward(gradient=gradient)
        #print("\n","Actor weight grad: ", self.l2.weight.grad)
        #print(,"Actor bias grad: ",self.l2.bias.grad)
        self.optimiser.step()
        self.optimiser.zero_grad()


    def small_weight_init(self,l):

        if isinstance(l,nn.Linear):
            #nn.init.normal_(l.weight,mean=0,std= 0.001)# std= 0.00005
            #l.bias.data.fill_(-2.5) # initialise to neg value so exp() return value < 1
            l.bias.data.fill_(-1) # initialise to neg value so exp() return value < 1
            l.weight.data.fill_(-1) # initialise to neg value so exp() return value < 1

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
