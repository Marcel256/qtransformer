from torch import nn

from qtransformer.logging.traín_logger import Logger


class Loss(nn.Module):
    logger: Logger

    def register_logger(self, logger):
        self.logger = logger

    def forward(self, states, actions, rewards, returns, terminals, timesteps):
        pass