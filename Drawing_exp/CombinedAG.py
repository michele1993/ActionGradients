import torch 

class CombActionGradient:

    def __init__(self, actor, action_s, rbl_weight=1, ebl_weight=1):


        self.actor = actor

        self.rbl_weight = torch.repeat_interleave(torch.tensor([rbl_weight]),2,dim=-1)
        self.ebl_weight = torch.repeat_interleave(torch.tensor([ebl_weight]),2,dim=-1)
    
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

        dr_dy = torch.autograd.grad(torch.sum(delta_rwd), y)[0] # take sum to compute grad across batch (it is okay since independent batches)
        # Compute grad relatice to mean and std of Gaussian policy
        dr_dy_dmu = torch.autograd.grad(est_y,mu_a,grad_outputs=dr_dy, retain_graph=True)[0] 
        dr_dy_dstd = torch.autograd.grad(est_y,std_a,grad_outputs=dr_dy, retain_graph=True)[0] 

        #Combine two grads relative to mu and std into one vector
        E_grad =  self.ebl_weight * torch.cat([dr_dy_dmu, dr_dy_dstd],dim=1)

        return E_grad

    def compute_dyda(self, y, action):





