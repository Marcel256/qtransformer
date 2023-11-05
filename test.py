import torch

out = torch.randn(8, 6, 3)
actions = torch.zeros((8,6, 1))

next_dim = torch.max(out[:, ], dim=2, keepdim=True)[0]
targets = torch.zeros_like(out)