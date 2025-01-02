from torch import nn

from qtransformer.train_logging.traín_logger import Logger


class Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def register_logger(self, logger):
        self.logger = logger

    def forward(self, states, actions, rewards, returns, terminals, timesteps):
        pass