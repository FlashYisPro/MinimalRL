#LIBRARIES
import torch 
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

#HELPERS
import os

#RL STUFF
from models.dqn import DQNLinear
from policy.base import BasePolicy
from common.utils import Softupdate

#######################
#DOUBLE DEEP Q-NETWORK.
#######################
#STATE-SPACE: Continuous
#ACTION-SPACE: Discrete
#OFF-POLICY

class DDQNPolicy(BasePolicy):
    def __init__(self,state_dim,n_actions,environment_name = "env",gamma = 0.99,tau = 0.999,
        learning_rate = 0.001,eps_decay = 0.995,eps_min = 0.1,):
        super().__init__(environment_name=environment_name,algorithm="DDQN_")

        #ENVIRONMENT PARAMETERS
        self.state_dim = state_dim
        self.n_actions = n_actions
    
        #ALGORITHM HYPERPARAMETERS 
        self.gamma = gamma
        self.tau = tau
        self.lr = learning_rate
        self.eps_decay = eps_decay
        self.eps_min = eps_min

        self.eps = 1.0
        self.noise = True

        self.critic = DQNLinear(self.state_dim,self.n_actions)
        self.critic_target = DQNLinear(self.state_dim,self.n_actions)
        self.critic_optim = optim.Adam(self.critic.parameters(),lr = self.lr)
        # self.critic_optim = optim.RMSprop(self.critic.parameters(),lr=self.lr)
        Softupdate(tau=0,target=self.critic_target,current=self.critic)

    def reset_eps(self):
        self.eps = 1.0

    def train_mode(self):
        self.noise = True

    def test_mode(self):
        self.noise = False

#GIVEN STATE OUTPUT AN ACTION ACCORDING TO CURRENT POLICY. RETURNS ACTION(Valid for gym).
    def act(self,state):
        if self.noise and np.random.rand()<=self.eps:
            action = np.random.choice(self.n_actions)
            return action
        
        state = torch.FloatTensor(state)
        with torch.no_grad():
            qvals = self.critic(state)
            action = torch.argmax(qvals,dim=0).item()

        return action

#UPDATE ACTOR AND CRITIC BASED ON SAMPLES DRAWN FROM ER BUFFER. RETURNS NOTHING.
    def learn(self,states,actions,rewards,next_states,dones):

        self.eps = max(self.eps_min,self.eps*self.eps_decay)

        currents = self.critic(states)
        targets = rewards + self.gamma*(1-dones)*self.critic_target(next_states).gather(1,torch.argmax(self.critic(next_states),dim=1).view(-1,1))

        currents = currents.gather(1,actions.long())

        batch_loss = F.mse_loss(currents,targets)

        self.critic_optim.zero_grad()
        batch_loss.backward()
        self.critic_optim.step()

        Softupdate(tau=self.tau,target=self.critic_target,current=self.critic)


    def save_models(self,name):
        path = self.save_location + str(self.current_loaded+int(name))
        torch.save(self.critic.state_dict(),path + "_Critic.pth")
        
    def load_models(self,name):
        self.current_loaded = int(name)
        path = self.save_location + name
        self.critic.load_state_dict(torch.load(path + "_Critic.pth"))
        Softupdate(tau=0,target=self.critic_target,current=self.critic)
        
####################################################################################################################################################