import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    if name == "ReLU":
        return nn.ReLU()
    elif name == "PReLU":
        return nn.PReLU()