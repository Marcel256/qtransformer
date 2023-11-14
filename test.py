import torch
from qtransformer import QTransformer
import numpy as np

out = torch.randn(8, 6, 3)
actions = torch.zeros((8,6, 1))

next_dim = torch.max(out[:, ], dim=2, keepdim=True)[0]
targets = torch.zeros_like(out)




model = QTransformer(17, 6, 256, 256, 4)



input_state = torch.randn((1, 4, 17))
input_action = torch.zeros((1, 6)).int()
model.eval()
print(model(input_state, input_action)[0])
input_action[0, 1] = 17
print(model(input_state, input_action)[0])