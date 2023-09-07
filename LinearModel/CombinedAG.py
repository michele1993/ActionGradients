import torch 

class CombActionGradient:

    def __init__(self, actor, beta_mu, beta_std, rbl_std_weight=1, ebl_std_weight=1):

        assert beta_mu >= 0 and beta_mu <= 1, "beta must be between 0 and 1 (inclusive)"

        self.actor = actor
        self.beta_mu = beta_mu
        self.beta_std = beta_std

        self.rbl_std_weight = rbl_std_weight
        self.ebl_std_weight = ebl_std_weight

        self.beta = torch.tensor([beta_mu, beta_std])
    
    def update(self, y, est_y, action, mu_a, std_a, delta_rwd):
        """ Perform update by comgining two gradient updates """

        R_grad = self.computeRBLGrad(action, mu_a, std_a, delta_rwd)
        E_grad = self.computeEBLGrad(y, est_y, action, mu_a, std_a, delta_rwd)

        comb_action_grad = self.beta * E_grad + (1-self.beta) * R_grad # Combine the two gradients

        action_variables = torch.stack([mu_a, std_a])
        agent_grad = self.actor.ActionGrad_update(comb_action_grad, action_variables)
    
    def computeRBLGrad(self, action, mu_a, std_a, delta_rwd):
        """ Compute reward-based learning (REINFORCE) action gradient 
        NOTE: Here we are computing the gradients explicitly, so need to specify torch.no_grad()
        """
        ##NOTE: want to min(delta_rwd) (since it is actually a punishment) so perform gradient descent with REIF grad
        with torch.no_grad():
            # Mean action grad 
            R_dr_dmu_a = (1/(std_a**2)) * (action - mu_a) * delta_rwd
            # Std action grad
            #R_dr_dstd_a = (delta_rwd * ((action - mu_a)**2 - std_a**2) / std_a**3)
            R_dr_dstd_a = self.rbl_std_weight * (delta_rwd * (action - mu_a)**2)

        #Combine two grads relative to mu and std into one vector
        R_grad = torch.cat([R_dr_dmu_a, R_dr_dstd_a])

        return R_grad
    
    def computeEBLGrad(self, y, est_y, action, mu_a, std_a, delta_rwd):
        """Compute error-based learning (MBDPG) action gradient 
        NOTE: torch.with_nograd not required here since autograd.grad does not compute grad by default 
        """ 
        dr_dy = torch.autograd.grad(delta_rwd, y)[0]
        # NOTE: here I diff relative to det_a instead of action, should be the same (since sigma is fixed)
        E_dr_dmu_a = torch.autograd.grad(est_y,mu_a,grad_outputs=dr_dy, retain_graph=True)[0] 

        ## NOTE: During error-based learning variance seems fixed!!! so not plausible the variance reduction
        #Error-based learning will try to converge to deterministic policy
        std_loss = std_a**2
        E_dr_dstd_a =  self.ebl_std_weight * torch.autograd.grad(std_loss, std_a, retain_graph=True)[0]

        # TRIAL: Error-based learning not controlling std
        #E_dr_dstd_a = torch.zeros_like(E_dr_dmu_a)

        #Combine two grads relative to mu and std into one vector
        E_grad = torch.stack([E_dr_dmu_a, E_dr_dstd_a])

        return E_grad
