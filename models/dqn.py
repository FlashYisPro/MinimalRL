import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import os

####################################################################################################################################################
#DEEP Q-NETWORK.
class DQNLinear(nn.Module):
    def __init__(self,state_dim,n_actions) -> None:
        super(DQNLinear,self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_dim,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,n_actions)
        )

    def forward(self,states):
        x = self.fc(states)
        return x
