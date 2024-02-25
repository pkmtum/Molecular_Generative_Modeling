from typing import Dict, Any, Tuple, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter

from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree, scatter


class ECCConv(MessagePassing):
    r""" The graph convolution operator from the 
    `"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs"
    <https://arxiv.org/abs/1704.02901>`_ paper """
    
    def __init__(self, num_edge_features: int, in_channels: int, out_channels: int):
        super().__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_fc = Linear(num_edge_features, in_channels * out_channels)
        self.bias = Parameter(torch.empty(out_channels))
        self.res_fc = Linear(in_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.edge_fc.reset_parameters()
        self.bias.data.zero_()
        self.res_fc.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        theta = self.edge_fc(edge_attr)
        theta = theta.view(-1, self.out_channels, self.in_channels)

        # normalization factor
        deg = degree(edge_index[0], x.size(0), dtype=x.dtype)
        deg_rcp = 1.0 / deg
        deg_rcp[deg_rcp == float("inf")] = 0
        norm = deg_rcp.unsqueeze(-1)
        
        # message passing
        out = self.propagate(edge_index, x=x, norm=norm, theta=theta)

        # add bias
        out += self.bias

        # identity connection
        out += self.res_fc(x)

        return out

    def message(self, x_j, norm_j, theta):
        return norm_j * torch.bmm(theta, x_j.unsqueeze(-1)).squeeze(-1)


class GlobalGraphPooling(nn.Module):
    r""" Graph-level output from equation 7 in the
    `"Gated Graph Sequence Neural Networks"
    <https://arxiv.org/abs/1511.05493>`_ paper
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.fc_i = Linear(in_channels, out_channels)
        self.fc_j = Linear(in_channels, out_channels)

    def forward(self, x: Tensor, batch: Optional[Tensor]) -> Tensor:
        attention_weights = F.sigmoid(self.fc_i(x))
        node_features = F.tanh(self.fc_j(x))

        out = attention_weights * node_features

        # sum all node features within a graph
        dim = -1 if isinstance(x, Tensor) and x.dim() == 1 else -2
        if batch is None:
            out = out.mean(dim=dim, keepdim=x.dim() <= 2)
        else:
            out = scatter(out, batch, dim=dim, reduce='sum')
        
        return F.tanh(out)


class Encoder(nn.Module):

    def __init__(self, hparams: Dict[str, Any]) -> None:
        super().__init__()

        self.latent_dim = hparams["latent_dim"]

        self.ecc_conv_1 = ECCConv(
            num_edge_features=hparams["num_edge_features"],
            in_channels=hparams["num_node_features"],
            out_channels=32
        )
        self.batch_norm_1 = BatchNorm(in_channels=32)
        self.relu_1 = nn.PReLU()
        self.ecc_conv_2 = ECCConv(
            num_edge_features=hparams["num_edge_features"],
            in_channels=32,
            out_channels=64
        )
        self.batch_norm_2 = BatchNorm(in_channels=64)
        self.relu_2 = nn.PReLU()
        self.ecc_conv_3 = ECCConv(
            num_edge_features=hparams["num_edge_features"],
            in_channels=64,
            out_channels=128
        )
        self.batch_norm_3 = BatchNorm(in_channels=128)
        self.relu_3 = nn.PReLU()

        self.graph_pooling = GlobalGraphPooling(in_channels=128, out_channels=128)

        # output 2 time the size of the latent vector
        # one half contains mu and the other half log(sigma)
        self.fc = nn.Linear(in_features=128, out_features=self.latent_dim * 2)


    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        x = self.ecc_conv_1(x, edge_index, edge_attr)
        x = self.batch_norm_1(x)
        x = self.relu_1(x)
        x = self.ecc_conv_2(x, edge_index, edge_attr)
        x = self.batch_norm_2(x)
        x = self.relu_2(x)
        x = self.ecc_conv_3(x, edge_index, edge_attr)
        x = self.batch_norm_3(x)
        x = self.relu_3(x)

        x = self.graph_pooling(x, batch)

        x = self.fc(x)

        mu = x[:, :self.latent_dim]
        log_sigma = x[:, self.latent_dim:]
        return mu, log_sigma