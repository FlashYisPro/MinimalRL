import gym
import warnings
from memory.discrete import UniformReplayBufferDiscrete
from policy.dqn import DQNPolicy
from trainers.offpolicy import OffPolicyTrainer

warnings.filterwarnings("ignore")

ENV = "CartPole-v1"
LR = 0.0003
GAMMA = 0.99
TAU = 0.999
BATCH_SIZE = 32
BUFFER_SIZE = 200000
EPOCHS = 50
EPOCH_TIMESTEPS = 5000
COLLECT_TIMESTEPS = 100

train_env = gym.make(ENV)
test_env = gym.make(ENV)

state_dim = train_env.observation_space.shape[0]
n_actions = train_env.action_space.n


replay_buffer = UniformReplayBufferDiscrete(BUFFER_SIZE,BATCH_SIZE,state_dim)
policy = DQNPolicy(state_dim,n_actions,ENV,GAMMA,TAU,LR)
trainer = OffPolicyTrainer(train_env,test_env,policy,replay_buffer,epochs=EPOCHS,epoch_timesteps=EPOCH_TIMESTEPS,collect_timesteps=COLLECT_TIMESTEPS)

trainer.train()