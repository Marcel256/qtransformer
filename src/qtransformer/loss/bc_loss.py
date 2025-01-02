from torch import nn
import torch

from qtransformer.loss.loss import Loss
from qtransformer.model.qtransformer import QTransformer


class BCLoss(Loss):

    def __init__(self, model: QTransformer):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()

    def forward(self, states, actions, rewards, returns, terminals, timesteps):
        logits = self.model(states[:, :-1], actions[:, -2], timesteps[:, :-1])
        logits = torch.swapaxes(logits, 1, 2)
        err = self.loss(logits, actions[:, -2])
        return err
