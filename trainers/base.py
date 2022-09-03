class BaseTrainer:
    def __init__(self,train_env,test_env) -> None:
        self.train_env = train_env
        self.test_env = test_env

    def train(self):
        raise NotImplementedError

    