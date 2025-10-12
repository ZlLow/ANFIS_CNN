import torch

Y_HORIZON = 13
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"