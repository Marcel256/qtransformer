from distutils.command.config import config

from qtransformer.evaluator.evaluator import Evaluator


import gym
import d4rl

import torch

import numpy as np
from qtransformer.env.seq_env_wrapper import SequenceEnvironmentWrapper
import random

from qtransformer.trainer_config import TrainerConfig

def transform_batched(history):
    states = [h['observations'] for h in history]
    return torch.from_numpy(np.stack(states)).float()

class D4RLEvaluator(Evaluator):
    def __init__(self, config: TrainerConfig):
        self.config = config
        random.seed(0)

    def evaluate(self, model, env_id, seq_len, episodes=100):
        envs = []
        returns = []
        history = []
        dones = []
        a_max = self.config.env_config.action_max
        a_min = self.config.env_config.action_min
        action_transform = lambda x: (x / self.config.model.action_bins) * (a_max - a_min) + a_min

        for ep in range(episodes):
            env = SequenceEnvironmentWrapper(gym.make(env_id), num_stack_frames=seq_len,
                                             action_dim=model.action_dim, action_transform=action_transform)
            history.append(env.reset(seed=random.randint(0, 2 ** 32 - 1)))
            returns.append(0)
            dones.append(False)
            envs.append(env)
        steps = 0
        while not all(dones):
            states = transform_batched(history)
            timesteps = torch.clip(torch.arange(steps - states.shape[1], steps), 0, 1000).unsqueeze(0)
            actions = model.predict_action(states, timesteps)
            for i in range(episodes):
                if not dones[i]:
                    next_state, reward, done, info = envs[i].step(actions[i])
                    history[i] = next_state
                    returns[i] += reward
                    dones[i] = done

            steps += 1

        return np.mean(returns)