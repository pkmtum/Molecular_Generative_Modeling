from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphAttentionPooling(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        # TODO:
        return x


class Encoder(nn.Module):

    def __init__(self, hparams: Dict[str, Any]) -> None:
        super().__init__()

        self.latent_dim = hparams["latent_dim"]

        # TODO: two graph convolutional layers (32 and 64 channels) with identity connection (edge conditioned graph convolution)
        self.conv1 = GCNConv(in_channels=hparams["num_node_features"], out_channels=32)
        self.conv2 = GCNConv(in_channels=32, out_channels=64)

        self.fc1 = nn.Linear(in_features=64, out_features=128)
        # output 2 time the size of the latent vector
        # one half contains mu and the other half log(sigma)
        self.fc2 = nn.Linear(in_features=128, out_features=self.latent_dim * 2)


    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        mu = x[:, :self.latent_dim]
        log_sigma = x[:, self.latent_dim:]
        return mu, log_sigma