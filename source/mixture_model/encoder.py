from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm

from graph_vae.encoder import ECGConv, GlobalGraphPooling
from .common import ResidualBlock


class MixtureModelEncoder(nn.Module):

    def __init__(self, hparams: Dict[str, Any]) -> None:
        super().__init__()

        self.z_latent_dim = hparams["z_dim"]

        channels = [16, 32, 64, 128]
        module_list = [
            ECGConv(
                num_edge_features=hparams["num_bond_types"],
                in_channels=hparams["num_atom_types"],
                out_channels=channels[0],
            ),
            BatchNorm(in_channels=channels[0]),
            nn.GELU()
        ]
        for i in range(len(channels) - 1):
            module_list.extend([
                ECGConv(
                    num_edge_features=hparams["num_bond_types"],
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                ),
                BatchNorm(in_channels=channels[i + 1]),
                nn.GELU()
            ])

        self.gnn_layers = nn.ModuleList(module_list)
        self.z_head = ResidualBlock(
            in_features=channels[-1],
            hidden_features=128,
            out_features=self.z_latent_dim * 2,
        )

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        for layer in self.gnn_layers:
            if isinstance(layer, ECGConv):
                x = layer(x, edge_index, edge_attr)
            else:
                x = layer(x)

        z = self.z_head(x)

        z_mu = z[:, :self.z_latent_dim]
        z_log_sigma = z[:, self.z_latent_dim:]
        z_sigma = torch.exp(torch.clamp(z_log_sigma, -30, 20))

        return z_mu, z_sigma