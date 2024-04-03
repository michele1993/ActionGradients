import torch 
import numpy as np

class CombActionGradient:

    def __init__(self, actor, action_s, rbl_weight=1, ebl_weight=1):
        
        assert np.isscalar(rbl_weight) and np.isscalar(ebl_weight), "ebl and rbl must be scalar"

        self.actor = actor

        # multiply action_dim *2 since have mean and std params
        self.rbl_weight = torch.repeat_interleave(torch.tensor([rbl_weight]),2*action_s,dim=-1)
        self.ebl_weight = torch.repeat_interleave(torch.tensor([ebl_weight]),2*action_s,dim=-1)
    
    def computeRBLGrad(self, action, mu_a, std_a, delta_rwd):
        """ Compute reward-based learning (REINFORCE) action gradient 
        NOTE: Here we are computing the gradients explicitly, so need to specify torch.no_grad()
        """
        ##NOTE: want to min(delta_rwd) (since it is actually a punishment) so perform gradient descent with REIF grad
        with torch.no_grad():
            # Mean action grad 
            R_dr_dmu_a = (1/(std_a**2)) * (action - mu_a) * delta_rwd
            # Std action grad
            R_dr_dstd_a = (delta_rwd * ((action - mu_a)**2 - std_a**2) / std_a**3)
            #R_dr_dstd_a = (delta_rwd * (action - mu_a)**2)

        #Combine two grads relative to mu and std into one vector
        R_grad =  self.rbl_weight * torch.cat([R_dr_dmu_a, R_dr_dstd_a],dim=1)
        

        return R_grad

    def computeEBLGrad(self, y, est_y, action, mu_a, std_a, delta_rwd):
        """Compute error-based learning (MBDPG) action gradient """

        dr_dy = torch.autograd.grad(torch.sum(delta_rwd), y, retain_graph=True)[0] # take sum to compute grad across batch (it is okay since independent batches)
        # Compute grad relatice to mean and std of Gaussian policy
        dr_dy_dmu = torch.autograd.grad(est_y,mu_a,grad_outputs=dr_dy, retain_graph=True)[0] 
        dr_dy_dstd = torch.autograd.grad(est_y,std_a,grad_outputs=dr_dy, retain_graph=True)[0] 

        #Combine two grads relative to mu and std into one vector
        E_grad =  self.ebl_weight * torch.cat([dr_dy_dmu, dr_dy_dstd],dim=1)

        return E_grad

    def compute_drdy(self, r, y):
        return torch.autograd.grad(torch.sum(r), y, retain_graph=True)[0] # take sum to compute grad across batch (it is okay since independent batches)

    def compute_dyda(self, y, x):
        x_s = x.size()[-1]
        batch_s, y_s = y.size()
        Jacobian = torch.zeros(batch_s, y_s, x_s)

        ## Compute Jacobian in naive way (i.e., using for loop) functorch API not so easily adaptable to 
        for i in range(y_s):
            Jacobian[:,i,:] = torch.autograd.grad(torch.sum(y[:,i]), x, retain_graph=True)[0]

        return Jacobian

    def compute_da_dmu(self, action, mu):
        x_s = mu.size()[-1]
        batch_s, a_s = action.size()
        Jacobian = torch.zeros(batch_s, a_s, x_s)
        ## Compute Jacobian in naive way (i.e., using for loop) functorch API not so easily adaptable to 
        for i in range(a_s):
            Jacobian[:,i,:] = torch.autograd.grad(torch.sum(action[:,i]), mu, retain_graph=True)[0]
        return Jacobian

    def compute_da_dstd(self, action, std):
        x_s = std.size()[-1]
        batch_s, a_s = action.size()
        Jacobian = torch.zeros(batch_s, a_s, x_s)
        ## Compute Jacobian in naive way (i.e., using for loop) functorch API not so easily adaptable to 
        for i in range(a_s):
            Jacobian[:,i,:] = torch.autograd.grad(torch.sum(action[:,i]), std, retain_graph=True)[0]
        return Jacobian

