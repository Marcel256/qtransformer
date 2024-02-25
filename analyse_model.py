import torch
from qtransformer import QTransformer
from sequence_dataset import SequenceDataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


model = QTransformer(17, 6, 512, 256, 4)
checkpoint = torch.load('models/model-1499.pt')

model.load_state_dict(checkpoint['model_state'])
model.eval()


dataset = SequenceDataset('data/random_seq.pkl', 4)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for batch in dataloader:
    states, actions, rewards, returns, terminal = batch

    q = torch.sigmoid(model(states[:, :-1].float(), actions[:, -2].long()))
    print(actions[:,-2])
    plt.bar(range(0, 256), q[0, 0].detach().numpy())
    plt.show()
    print(model.predict_action(states[:, :-1].float())[0])
    break