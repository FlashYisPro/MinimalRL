from trainers.base import BaseTrainer
from policy.base import BasePolicy
from memory.base import BaseReplayBuffer
from common.utils import GenericPlot
from evaluators.evaluate import Evaluator
from tqdm import tqdm

class OffPolicyTrainer(BaseTrainer):
    def __init__(self,train_env,test_env,policy: BasePolicy,buffer: BaseReplayBuffer, epochs = 10,
        epoch_timesteps = 5000, stop_fn = lambda x : False, eval_episodes = 10, learn_fraction = 1.0,
        collect_timesteps = 100,save_checkpoints = True, plot_results = True):

        super().__init__(train_env,test_env)
        self.buffer = buffer
        self.policy = policy
        
        self.eval_episodes = eval_episodes
        self.stop_fn = stop_fn
        self.epochs = epochs
        self.epoch_timesteps = epoch_timesteps
        self.collect_timesteps = collect_timesteps
        self.learn_fraction = learn_fraction

        self.n_miniepoch = int(self.epoch_timesteps/self.collect_timesteps)

        self.save_checkpoints = save_checkpoints
        self.plot_results = plot_results

        self.evaluator = Evaluator(self.test_env,self.policy)

    def train(self):
        rewards_array = []
        time_array = []

        #INITIAL REWARD CALC BEFORE TRAINING
        rewards_array.append(self.evaluator.get_average_reward(episodes=self.eval_episodes))
        time_array.append(0)

        for epoch in range(1,self.epochs+1):
            print(f"***************** EPOCH: {epoch} *****************")
            
            #Put policy in training mode.
            self.policy.train_mode()
            
            #Train the policy and collect new transitions
            self.run_epoch()

            #Evaluate after training and show results
            curr_reward = self.evaluator.get_average_reward(episodes=self.eval_episodes)
            print(f"Average Reward: {curr_reward}")

            #Checkpointing
            if self.save_checkpoints:
                self.policy.save_models(str(epoch))

            #Plotting Results
            if self.plot_results:
                rewards_array.append(curr_reward)
                time_array.append(epoch)
                GenericPlot([(time_array,rewards_array)],self.policy.results_location+self.policy.results_name,xlabel="Epochs",
                ylabel= "Average Return")
                
            #Early Stopping
            if self.stop_fn(curr_reward):
                print("Stop function returned True")
                break

        return rewards_array,time_array


    def run_epoch(self):
        done = True
        for step in tqdm(range(self.epoch_timesteps)):
            if done:
                state = self.train_env.reset()
                done = False

            action = self.policy.act(state)
            next_state,reward,done,_ = self.train_env.step(action)

            self.buffer.add_transition(state,action,reward,next_state,done)

            state= next_state

            if self.buffer.len > self.buffer.batch_size:
                states,actions,rewards,next_states,dones = self.buffer.sample_minibatch()
                self.policy.learn(states,actions,rewards,next_states,dones)


#THESE FUNCTIONS ARE FOR FUTURE USE.
#THEY DO NOT WORK AS GOOD IN TERMS OF TRAINING.
#BUT THEY PERFORM REALLY WELL WITH RESPECT TO TRAINING TIME.
    def collect_steps(self,steps):
        done = True

        for step in range(steps):
            if done:
                state = self.train_env.reset()
                done = False

            action = self.policy.act(state)
            next_state,reward,done,_ = self.train_env.step(action)
            
            self.buffer.add_transition(state,action,reward,next_state,done)
            
            state = next_state

    def learn_steps(self,steps):
        for step in range(steps):
            if self.buffer.len > self.buffer.batch_size:
                states,actions,rewards,next_states,dones = self.buffer.sample_minibatch()
                self.policy.learn(states,actions,rewards,next_states,dones)

        
