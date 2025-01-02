from collections import deque

from qtransformer.evaluator.evaluator import Evaluator
from qtransformer.train_logging.wandb_logger import WandbLogger
from qtransformer.loss.loss import Loss
from qtransformer.model.qtransformer import QTransformer
from qtransformer.trainer_config import TrainerConfig
from qtransformer.util.schedulers import get_cosine_schedule_with_warmup

import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv

import numpy as np
import os

load_dotenv()

class Trainer:

    def __init__(self, model: QTransformer, config: TrainerConfig, loss:  Loss, evaluator: Evaluator, dataloader: DataLoader, device):
        self.model = model
        self.config = config
        self.evaluator = evaluator
        self.dataloader = dataloader
        self.loss = loss
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.train_config.lr)

        self.logging_steps = 50
        self.evaluation_steps = 5000

        self.losses = deque(maxlen=50)
        self.logger = WandbLogger(os.environ['WANDB_ENTITY'], os.environ['WANDB_PROJECT'], config)
        self.loss.register_logger(self.logger)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, int(0.1*config.train_config.train_steps), config.train_config.train_steps)


    def train(self):
        total_steps = self.config.train_config.train_steps
        seq_len = self.config.model.seq_len
        i = 0
        while i < total_steps:
            for batch in self.dataloader:
                states, actions, rewards, returns, terminal, timesteps = batch

                states = states.float().to(self.device)
                actions = torch.reshape(actions, (actions.shape[0], seq_len + 1, -1)).long().to(self.device)
                returns = returns.float().to(self.device)
                rewards = rewards.float().to(self.device)
                timesteps = timesteps.int().to(self.device)
                terminal = terminal.unsqueeze(2).float().to(self.device)

                err = self.loss(states, actions, rewards, returns, terminal, timesteps)

                self.optimizer.zero_grad()
                err.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train_config.max_grad_norm)
                self.optimizer.step()
                self.losses.append(err.item())
                self.scheduler.step()

                if i % self.evaluation_steps == 0:
                    self.model.eval()
                    score = self.evaluator.evaluate(self.model, self.config.env_config.id, self.config.model.seq_len)
                    self.logger.log_metrics({"eval_score": score})
                    self.model.train()

                if i % self.logging_steps == 0:
                    self.logger.log_metrics({"train_loss": np.mean(self.losses)})
                    self.logger.write_step()
                if i >= total_steps:
                    break
                i += 1
        
        self.model.eval()
        score = self.evaluator.evaluate(self.model, self.config.env_config.id, self.config.model.seq_len)
        self.logger.log_metrics({"eval_score": score})
        self.logger.write_step()
        print("Final Score: ", score)