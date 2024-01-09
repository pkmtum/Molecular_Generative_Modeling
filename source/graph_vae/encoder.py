from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from torch_geometric.nn.conv import MessagePassing

class ECCConv(MessagePassing):
    r""" The graph convolution operator from the 
    `"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs"
    <https://arxiv.org/abs/1609.02907>`_ paper """
    
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')

    def forward(self, x, ) -> Any:
        # TODO
        pass

class GraphAttentionPooling(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, edge_index, edge_attr):
        # TODO:
        return x


class Encoder(nn.Module):

    def __init__(self, hparams: Dict[str, Any]) -> None:
        super().__init__()

        self.latent_dim = hparams["latent_dim"]

        # TODO: two graph convolutional layers (32 and 64 channels) with identity connection (edge conditioned graph convolution)
        self.conv_1 = GCNConv(in_channels=hparams["num_node_features"], out_channels=32)
        self.batch_norm_1 = BatchNorm(in_channels=32)
        self.conv_2 = GCNConv(in_channels=32, out_channels=64)
        #self.batch_norm_2 = BatchNorm(in_channels=64)

        self.fc_1 = nn.Linear(in_features=64, out_features=128)
        # output 2 time the size of the latent vector
        # one half contains mu and the other half log(sigma)
        self.fc_2 = nn.Linear(in_features=128, out_features=self.latent_dim * 2)


    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        x = self.conv_1(x, edge_index)
        x = self.batch_norm_1(x)
        x = F.relu(x)
        x = self.conv_2(x, edge_index) 

        x = global_mean_pool(x, batch)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)

        mu = x[:, :self.latent_dim]
        log_sigma = x[:, self.latent_dim:]
        return mu, log_sigma