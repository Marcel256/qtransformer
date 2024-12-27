import gym
import d4rl

import torch

import numpy as np
from seq_env_wrapper import SequenceEnvironmentWrapper
import random

def transform_batched(history):
    states = [h['observations'] for h in history]
    return torch.from_numpy(np.stack(states)).float()
def batched_eval(env_name, model, episodes, num_stack_frames=4, action_dim=1, action_transform=None):
    envs = []
    returns = []
    history = []
    dones = []
    for ep in range(episodes):
        env = SequenceEnvironmentWrapper(gym.make(env_name), num_stack_frames=num_stack_frames, action_dim=action_dim, action_transform=action_transform)
        history.append(env.reset(seed=random.randint(0, 2**32 - 1)))
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