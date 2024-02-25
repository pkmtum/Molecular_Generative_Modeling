from typing import Dict, Any, Tuple

import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self, hparams: Dict[str, Any]) -> None:
        super().__init__()

        self.fcls = nn.Sequential(
            nn.Linear(in_features=hparams["latent_dim"], out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.PReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.PReLU(),
        )

        self.max_num_nodes = hparams["max_num_nodes"] 
        self.num_node_features = hparams["num_node_features"]

        # the atom graph is symmetric so we only predict the upper triangular part
        # and the diagonal that indicates the presence of nodes
        upper_triangular_diag_size = int(self.max_num_nodes * (self.max_num_nodes + 1) / 2)
        self.fc_adjacency = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(in_features=512, out_features=upper_triangular_diag_size)
        )

        self.fc_node_features = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(in_features=512, out_features=self.max_num_nodes * self.num_node_features),
        )
        
        self.max_num_edges = int(self.max_num_nodes * (self.max_num_nodes - 1) / 2)
        self.num_edge_features = hparams["num_edge_features"]
        self.fc_edge_features = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(in_features=512, out_features=self.max_num_edges * self.num_edge_features)
        )
        

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.fcls(z)
        # predict upper triangular matrix including the diagonal
        adj_triu_mat = self.fc_adjacency(x)
        node_features = self.fc_node_features(x)
        edge_features = self.fc_edge_features(x)

        # reshape matrices
        node_mat = node_features.view(-1, self.max_num_nodes, self.num_node_features)
        edge_triu_mat = edge_features.view(-1, self.max_num_edges, self.num_edge_features)

        return adj_triu_mat, node_mat, edge_triu_mat