import os
import torch
import numpy as np
from common.defaults import DATA_LOCATION

class OfflineBufferDiscrete:
    def __init__(self,state_dim) -> None:
        self.state_dim = state_dim
        self.len = 0
        self.clear_buffer()

    def clear_buffer(self):
        self.states = torch.zeros(0,self.state_dim,dtype=torch.float32)
        self.next_states = torch.zeros(0,self.state_dim,dtype=torch.float32)
        self.actions = torch.zeros(0,1,dtype=torch.int64)
        self.rewards = torch.zeros(0,1,dtype=torch.float32)
        self.dones = torch.zeros(0,1,dtype=torch.int8)

        self.len = 0

    def load_data(self,path):
        path = DATA_LOCATION + path
        if not os.path.exists(path):
            print("InvalidPath...")
            return

        states = torch.load(path+'states.tensor')
        actions = torch.load(path+'actions.tensor')
        rewards = torch.load(path+'rewards.tensor')
        next_states = torch.load(path+'next_states.tensor')
        dones = torch.load(path+'dones.tensor')

        loaded_size = states.shape[0]
        print(f"{loaded_size} transitions loaded!")

        self.states = torch.cat((self.states,states))
        self.next_states = torch.cat((self.next_states,next_states))
        self.actions = torch.cat((self.actions,actions))
        self.rewards = torch.cat((self.rewards,rewards))
        self.dones = torch.cat((self.dones,dones))

        self.actions_one_hot = torch.nn.functional.one_hot(self.actions.view(-1),torch.max(self.actions)+1)

        self.len += loaded_size
        print(f"Current Transitions in Buffer: {self.len}")
