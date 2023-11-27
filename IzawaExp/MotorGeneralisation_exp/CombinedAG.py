import torch 

class CombActionGradient:

    def __init__(self, actor, beta, rbl_weight=1, ebl_weight=1):

        assert beta >= 0 and beta <= 1, "beta must be between 0 and 1 (inclusive)"

        self.actor = actor
        self.beta = beta

        self.rbl_weight = torch.tensor(rbl_weight).unsqueeze(0)
        self.ebl_weight = torch.tensor(ebl_weight).unsqueeze(0)
    
    def update(self, y, est_y, action, mu_a, std_a, delta_rwd):
        """ Perform update by comgining two gradient updates """

        R_grad = self.computeRBLGrad(action, mu_a, std_a, delta_rwd)
        E_grad = self.computeEBLGrad(y, est_y, action, mu_a, std_a, delta_rwd)

        R_grad_norm = torch.norm(R_grad, dim=-1, keepdim=True)
        E_grad_norm = torch.norm(E_grad, dim=-1, keepdim=True)

        comb_action_grad = self.beta * (self.ebl_weight * E_grad/E_grad_norm) + (1-self.beta) * (self.rbl_weight*R_grad/R_grad_norm) # Combine the two gradients

        comb_action_grad *= self.beta * E_grad_norm + (1-self.beta) * R_grad_norm

        action_variables = torch.cat([mu_a, std_a],dim=-1)
        agent_grad = self.actor.ActionGrad_update(comb_action_grad, action_variables)

        return comb_action_grad
    
    def computeRBLGrad(self, action, mu_a, std_a, delta_rwd):
        """ Compute reward-based learning (REINFORCE) action gradient 
        NOTE: Here we are computing the gradients explicitly, so need to specify torch.no_grad()
        """

        with torch.no_grad():
            # Mean action grad 
            R_dr_dmu_a = (1/(std_a**2)) * (action - mu_a) * delta_rwd
            # Std action grad
            #R_dr_dstd_a = (delta_rwd * ((action - mu_a)**2 - std_a**2) / std_a**3)
            R_dr_dstd_a = (delta_rwd * (action - mu_a)**2)

        #Combine two grads relative to mu and std into one vector
        R_grad = torch.cat([R_dr_dmu_a, R_dr_dstd_a],dim=-1)

        return R_grad 
    
    def computeEBLGrad(self, y, est_y, action, mu_a, std_a, delta_rwd):
        """Compute error-based learning (MBDPG) action gradient 
        NOTE: torch.with_nograd not required here since autograd.grad does not compute grad by default 
        """ 
        # Take sum to compute grad across batch (it is okay since independent batches)
        dr_dy = torch.autograd.grad(torch.sum(delta_rwd), y, retain_graph=True)[0] 

        # NOTE: here I diff relative to det_a instead of action, should be the same (since sigma is fixed)
        E_dr_dmu_a = torch.autograd.grad(est_y,mu_a,grad_outputs=dr_dy, retain_graph=True)[0] 
        E_dr_dstd_a = torch.autograd.grad(est_y,std_a,grad_outputs=dr_dy, retain_graph=True)[0] 

        #Combine two grads relative to mu and std into one vector
        E_grad = torch.cat([E_dr_dmu_a, E_dr_dstd_a],dim=-1)

        return E_grad 
