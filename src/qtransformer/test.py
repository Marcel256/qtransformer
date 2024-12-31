from qtransformer.data.sequence_dataset import SequenceDataset

import numpy as np

obs1 = np.arange(5)
obs = np.concat([obs1 for _ in range(10)], axis=0)
terminal = np.zeros_like(obs)
terminal[obs == 4] = 1

actions = np.zeros_like(terminal)
rewards = np.zeros_like(terminal)
returns = np.zeros_like(terminal)


data = {
    "observations": obs,
    "actions": actions,
    "rewards": rewards,
    "terminals": terminal,
    "timesteps": obs,
    "returns": returns,
}

dataset = SequenceDataset(data, 2)

for i in range(len(dataset)):
    print(dataset[i][0])


