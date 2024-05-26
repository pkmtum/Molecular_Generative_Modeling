from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm

from graph_vae.encoder import ECCConv, GlobalGraphPooling


class MixtureModelEncoder(nn.Module):

    def __init__(self, hparams: Dict[str, Any]) -> None:
        super().__init__()

        self.z_latent_dim = hparams["z_dim"]
        self.eta_latent_dim = hparams["eta_dim"]

        channels = [16, 32, 64, 128]

        module_list = [
            ECCConv(
                num_edge_features=hparams["num_bond_types"],
                in_channels=hparams["num_atom_types"],
                out_channels=channels[0]
            ),
            BatchNorm(in_channels=channels[0]),
            nn.PReLU()
        ]
        for i in range(len(channels) - 1):
            module_list.extend([
                ECCConv(
                    num_edge_features=hparams["num_bond_types"],
                    in_channels=channels[i],
                    out_channels=channels[i + 1]
                ),
                BatchNorm(in_channels=channels[i + 1]),
                nn.PReLU()
            ])

        self.gnn_layers = nn.ModuleList(module_list)
        self.z_head = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Linear(128, self.z_latent_dim * 2)
        )

        self.graph_pooling = GlobalGraphPooling(in_channels=channels[-1], out_channels=channels[-1])
        self.eta_head = nn.Linear(in_features=channels[-1], out_features=self.eta_latent_dim * 2)
        

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        for layer in self.gnn_layers:
            if isinstance(layer, ECCConv):
                x = layer(x, edge_index, edge_attr)
            else:
                x = layer(x)

        z = self.z_head(x)

        z_mu = z[:, :self.z_latent_dim]
        z_log_sigma = z[:, self.z_latent_dim:]
        z_sigma = torch.exp(torch.clamp(z_log_sigma, -30, 20))

        eta = self.eta_head(self.graph_pooling(x, batch))

        eta_mu = eta[:, :self.eta_latent_dim]
        eta_log_sigma = eta[:, self.eta_latent_dim:]
        eta_sigma = torch.exp(torch.clamp(eta_log_sigma, -30, 20))

        return z_mu, z_sigma, eta_mu, eta_sigma