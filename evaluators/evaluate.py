from email import policy
from common.utils import GenericPlot
from policy.base import BasePolicy

class Evaluator:
    def __init__(self,env,policy: BasePolicy) -> None:
        self.env = env
        self.policy = policy

    def get_average_reward(self,episodes = 10):
        
        self.policy.test_mode()
        average_reward = 0
        

        for episode in range(episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = self.policy.act(state)
                next_state,reward,done,_ = self.env.step(action)
                average_reward += reward
                state = next_state

        average_reward /= episodes
        return average_reward


    def render_episode(self):
        
        self.policy.test_mode()
        state = self.env.reset()
        done = False
        total_reward = 0


        while not done:
            self.env.render(mode="human")
            action = self.policy.act(state)
            next_state,reward,done,_ = self.env.step(action)
            total_reward += reward
            state = next_state

        print(f"Total Reward in the episode is {total_reward}")
        return total_reward
