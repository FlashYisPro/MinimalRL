#HELPERS
import os

#RL STUFF
from common.defaults import RESULTS_LOCATION,CHECKPOINTS_LOCATION


class BasePolicy:
    def __init__(self,environment_name,algorithm):
        self.env_name = environment_name
        self.algorithm = algorithm

        #CHECKPOINTING AND RESULTS HANDLING
        self.save_location = CHECKPOINTS_LOCATION + self.algorithm[:-1] + "/" + self.env_name + "/"
        self.results_location = RESULTS_LOCATION + self.algorithm[:-1] + "/" + self.env_name + "/"
        self.results_name = self.algorithm + self.env_name

        if not os.path.exists(self.save_location):
            os.makedirs(self.save_location)
        
        if not os.path.exists(self.results_location):
            os.makedirs(self.results_location)

        self.current_loaded = 0

    def train_mode(self):
        raise NotImplementedError

    def test_mode(self):
        raise NotImplementedError

    def act(self):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def save_models(self):
        raise NotImplementedError

    def load_models(self):
        raise NotImplementedError