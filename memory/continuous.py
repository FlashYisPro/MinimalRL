import torch
import numpy as np
from memory.base import BaseReplayBufferContinuous

class UniformReplayBufferContinuous(BaseReplayBufferContinuous):
    def __init__(self, buffer_size, batch_size, state_dim, action_dim) -> None:
        super().__init__(buffer_size, batch_size, state_dim, action_dim)

    def sample_minibatch(self,batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.len <= batch_size:
            return None

        indices = np.random.choice(self.len,batch_size)

        states_batch = self.states[indices]
        actions_batch = self.actions[indices]
        rewards_batch = self.rewards[indices]
        next_states_batch = self.next_states[indices]
        dones_batch = self.dones[indices]

        return states_batch,actions_batch,rewards_batch,next_states_batch,dones_batch

    
        