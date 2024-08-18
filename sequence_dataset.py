import numpy as np
from torch.utils.data import Dataset
import pickle

def convert_action(action, a_min, a_max, action_bins):
  norm = (action - a_min)/(a_max-a_min)

  return np.rint(norm * action_bins).astype(np.uint8)

class SequenceDataset(Dataset):


    def __init__(self, file, seq_length):
        with open(file, 'rb') as fh:
            data = pickle.load(fh)

        self.__init__(data, seq_length)
    def __init__(self, data, seq_length):

        #-1 for zero padding at the end
        n_samples = data['observations'].shape[0] - 1


        self.obs = data['observations']
        self.action = data['actions']
        self.rewards = data['rewards']
        self.returns = data['returns']
        self.terminals = data['terminals']
        self.timesteps = data['timesteps']

        self.done_idx = np.where(self.terminals == 1)[0]
        self.seq_length = seq_length
        self.idx_list = []

        j = 0
        curr_end = self.done_idx[j] - seq_length + 1
        i = 0
        while i <= n_samples - seq_length:
            if i <= curr_end:
                self.idx_list.append(i)
            else:
                j += 1
                i = curr_end + seq_length - 1
                if j < len(self.done_idx):
                    curr_end = self.done_idx[j] - seq_length + 1
                else:
                    curr_end =  n_samples - seq_length

            i += 1

    def __getitem__(self, item):
        idx = self.idx_list[item]
        end = idx + self.seq_length + 1

        return self.obs[idx:end], self.action[idx: end], self.rewards[idx:end], self.returns[idx:end], self.terminals[idx:end], self.timesteps[idx:end]

    def __len__(self):
        return len(self.idx_list)


    @classmethod
    def from_d4rl(cls, dataset, seq_len, action_bins, gamma=0.99):
        data = dict()
        data['observations'] = dataset['observations']
        r = dataset['rewards']
        actions = convert_action(dataset['actions'], -1, 1, action_bins)
        terminal = dataset['timeouts']
        ret = np.zeros_like(r)
        timesteps = np.zeros_like(r)
        ret[-1] = r[-1]
        for i in reversed(range(r.shape[0] - 1)):
            if terminal[i]:
                ret[i] = r[i]
            else:
                ret[i] = r[i] + gamma * ret[i + 1]
        t = 0
        for i in range(r.shape):
            if terminal[i]:
                timesteps[i] = t
                t = 0
            else:
                timesteps[i] = t
                t += 1
        data['timesteps'] = timesteps
        data['returns'] = ret
        data['actions'] = np.clip(actions, 0, action_bins-1)
        data['rewards'] = r
        data['terminals'] = terminal

        return SequenceDataset(data, seq_len)

    @classmethod
    def from_rollouts(cls, dataset, seq_len, gamma=0.99):
        data = dict()
        data['observations'] = dataset['observations']
        r = dataset['rewards']
        actions = dataset['actions']
        terminal = dataset['terminals']
        ret = np.zeros_like(r)
        ret[-1] = r[-1]
        for i in reversed(range(r.shape[0] - 1)):
            if terminal[i]:
                ret[i] = r[i]
            else:
                ret[i] = r[i] + gamma * ret[i + 1]

        data['returns'] = ret
        data['actions'] = actions
        data['rewards'] = r
        data['terminals'] = terminal

        return SequenceDataset(data, seq_len)
