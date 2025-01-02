from qtransformer.evaluator.evaluator import Evaluator
from qtransformer.env.seq_env_wrapper_gymnasium import SequenceEnvironmentWrapper

import gymnasium as gym
import torch
from gymnasium.wrappers import AtariPreprocessing
import numpy as np
import random

def transform_batched(history):
    states = [h['observations'] for h in history]
    return torch.from_numpy(np.stack(states)).float() / 255.0

class AtariEvaluator(Evaluator):
    def __init__(self):
        pass

    def evaluate(self,  model, env_id, seq_len, episodes=100):
        envs = []
        returns = []
        history = []
        dones = []
        action_transform = lambda x: x[0]
        for ep in range(episodes):
            atari_env = AtariPreprocessing(gym.make(env_id, frameskip=1))
            env = SequenceEnvironmentWrapper(atari_env, num_stack_frames=seq_len, action_dim=model.action_dim,
                                             action_transform=action_transform)
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