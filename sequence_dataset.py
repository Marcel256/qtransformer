import numpy as np
from torch.utils.data import Dataset
import pickle


class SequenceDataset(Dataset):

    def __init__(self, file, seq_length):

        with open(file, 'rb') as fh:
            data = pickle.load(fh)

        #-1 for zero padding at the end
        n_samples = data['observations'].shape[0] - 1


        self.obs = data['observations']
        self.action = data['actions']
        self.rewards = data['rewards']
        self.returns = data['returns']
        self.terminals = data['terminals']

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

        return self.obs[idx:end], self.action[idx: end], self.rewards[idx:end], self.returns[idx:end], self.terminals[idx:end]

    def __len__(self):
        return len(self.idx_list)
