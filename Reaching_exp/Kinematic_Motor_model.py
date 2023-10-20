import torch
import numpy as np
import torch.nn as nn
import torch.optim as opt

class Kinematic_model():
    """ Implementation of a 2D kinematic model """

    def __init__(self,L=1.8):
        """Args:
            L: person height in meters
        """    
        # Based on standard measures
        self.l1 = L * 0.186 # lenght of upper arm (in meters)
        self.l2 = (0.146 + 0.108) * L # lenght of forearm (in meters)

    def step(self,action):
        """ Compute xy coordinates based on shoulder and elbow angle """
        x = self.l1 * torch.cos(action[...,0:1]) + self.l2 * torch.cos(action[...,0:1] + action[...,1:2])
        y = self.l1 * torch.sin(action[...,0:1]) + self.l2 * torch.sin(action[...,0:1] + action[...,1:2])
        return x,y
