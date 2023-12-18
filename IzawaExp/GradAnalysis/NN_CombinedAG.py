import torch 

class CombActionGradient:

    def __init__(self, actor, beta, rbl_weight=1, ebl_weight=1):

        assert beta >= 0 and beta <= 1, "beta must be between 0 and 1 (inclusive)"

        self.actor = actor
        self.beta = beta

        self.rbl_weight = torch.tensor(rbl_weight).unsqueeze(0)
        self.ebl_weight = torch.tensor(ebl_weight).unsqueeze(0)
    
    def computeRBLGrad(self, action, mu_a, std_a, delta_rwd):
        """ Compute reward-based learning (REINFORCE) action gradient 
        NOTE: Here we are computing the gradients explicitly, so need to specify torch.no_grad()
        """

        with torch.no_grad():
            # Mean action grad 
            R_dr_dmu_a = (1/(std_a**2)) * (action - mu_a) * delta_rwd
            R_dr_dstd_a = (delta_rwd * ((action - mu_a)**2 - std_a**2) / std_a**3)
        R_grad =  self.rbl_weight * torch.cat([R_dr_dmu_a, R_dr_dstd_a],dim=1)
        return R_grad 
    
    def computeEBLGrad(self, y, est_y, action, mu_a, std_a, delta_rwd):
        """Compute error-based learning (MBDPG) action gradient 
        NOTE: torch.with_nograd not required here since autograd.grad does not compute grad by default 
        """ 
        # Take sum to compute grad across batch (it is okay since independent batches)
        dr_dy = torch.autograd.grad(torch.sum(delta_rwd), y, retain_graph=True)[0] 

        # NOTE: here I diff relative to det_a instead of action, should be the same (since sigma is fixed)
        dr_dy_dmu = torch.autograd.grad(est_y,mu_a,grad_outputs=dr_dy, retain_graph=True)[0] 
        dr_dy_dstd = torch.autograd.grad(est_y,std_a,grad_outputs=dr_dy, retain_graph=True)[0] 

        #Combine two grads relative to mu and std into one vector
        E_grad =  self.ebl_weight * torch.cat([dr_dy_dmu, dr_dy_dstd],dim=1)
        return E_grad 
