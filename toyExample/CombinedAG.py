import torch 

class CombActionGradient:

    def __init__(self, actor, reinf_std_w, MBDPG_std_w, beta):

        assert beta >= 0 and beta <= 1, "beta must be between 0 and 1 (inclusive)"

        self.actor = actor
        self.reinf_std_w = reinf_std_w
        self.MBDPG_std_w = MBDPG_std_w
        self.beta = beta
    
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
            R_dr_dstd_a = self.reinf_std_w * (-1) * (delta_rwd * (std_a**2 - (action - mu_a)**2) / std_a**3)

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

        #Error-based learning will try to converge to deterministic policy
        std_loss = std_a**2
        E_dr_dstd_a = self.MBDPG_std_w * torch.autograd.grad(std_loss, std_a, retain_graph=True)[0]

        #Combine two grads relative to mu and std into one vector
        E_grad = torch.stack([E_dr_dmu_a, E_dr_dstd_a])

        return E_grad
