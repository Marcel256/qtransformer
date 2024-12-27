import gzip
import numpy as np
import os

def load_atari_dataset(path: str) -> dict:
    obs = np.load(gzip.GzipFile(filename=os.path.join(path, "obs.gz")))
    actions = np.load(gzip.GzipFile(filename=os.path.join(path, "action.gz")))
    rewards = np.load(gzip.GzipFile(filename=os.path.join(path, "reward.gz")))
    terminal = np.load(gzip.GzipFile(filename=os.path.join(path, "terminal.gz")))

    return dict(observations=obs, actions=actions, rewards=rewards, terminals=terminal, timeouts=np.zeros_like(terminal))


