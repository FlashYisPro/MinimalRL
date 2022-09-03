import torch
import os
from common.defaults import DATA_LOCATION


class BaseReplayBuffer:
    def __init__(self,buffer_size,batch_size,state_dim) -> None:
        #PARAMETERS
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_dim = state_dim

        #COUNTERS
        self.front_ptr = -1
        self.len = 0

    def save_buffers(self,path):
        print(f"Current Transitions: {self.len}")
        path = DATA_LOCATION + path
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(path+'states.tensor',self.states)
        torch.save(path+'next_states.tensor',self.next_states)
        torch.save(path+'actions.tensor',self.actions)
        torch.save(path+'rewards.tensor',self.rewards)
        torch.save(path+'dones.tensor',self.dones)

    def add_transition(self):
        raise NotImplementedError

    def sample_minibatch(self):
        raise NotImplementedError
        



class BaseReplayBufferDiscrete(BaseReplayBuffer):
    def __init__(self,buffer_size,batch_size,state_dim) -> None:
        super().__init__(buffer_size,batch_size,state_dim)

        #ACTUAL BUFFERS

        self.states = torch.zeros(self.buffer_size,self.state_dim,dtype=torch.float32)
        self.next_states = torch.zeros(self.buffer_size,self.state_dim,dtype=torch.float32)
        self.actions = torch.zeros(self.buffer_size,1,dtype=torch.int32)
        self.rewards = torch.zeros(self.buffer_size,1,dtype=torch.float32)
        self.dones = torch.zeros(self.buffer_size,1,dtype=torch.int8)


    def add_transition(self,state,action,reward,next_state,done):
        self.front_ptr += 1
        self.front_ptr %= self.buffer_size

        self.states[self.front_ptr] = torch.FloatTensor(state)
        self.next_states[self.front_ptr] = torch.FloatTensor(next_state)
        self.actions[self.front_ptr][0] = action
        self.rewards[self.front_ptr][0] = reward
        self.dones[self.front_ptr][0] = done

        self.len = self.len + 1 if self.len < self.buffer_size else self.len


class BaseReplayBufferContinuous(BaseReplayBuffer):
    def __init__(self,buffer_size,batch_size,state_dim,action_dim) -> None:
        super().__init__(buffer_size,batch_size,state_dim)
        self.action_dim = action_dim

        #ACTUAL BUFFERS
        self.states = torch.zeros(self.buffer_size,self.state_dim,dtype=torch.float32)
        self.next_states = torch.zeros(self.buffer_size,self.state_dim,dtype=torch.float32)
        self.actions = torch.zeros(self.buffer_size,self.action_dim,dtype=torch.float32)
        self.rewards = torch.zeros(self.buffer_size,1,dtype=torch.float32)
        self.dones = torch.zeros(self.buffer_size,1,dtype=torch.int8)
        


    def add_transition(self,state,action,reward,next_state,done):
        self.front_ptr += 1
        self.front_ptr %= self.buffer_size

        self.states[self.front_ptr] = torch.FloatTensor(state)
        self.next_states[self.front_ptr] = torch.FloatTensor(next_state)
        self.actions[self.front_ptr] = torch.FloatTensor(action)
        self.rewards[self.front_ptr][0] = reward
        self.dones[self.front_ptr][0] = done

        self.len = self.len + 1 if self.len < self.buffer_size else self.len