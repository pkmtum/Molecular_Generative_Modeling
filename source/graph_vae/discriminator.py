from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphDiscriminator(nn.Module):

    def __init__(self, hparams: Dict[str, Any]) -> None:
        super().__init__()

        self.max_num_nodes = hparams["max_num_nodes"]
        self.num_node_features = hparams["num_node_features"]

        # the atom graph is symmetric so we only predict the upper triangular part
        # and the diagonal that indicates the presence of nodes
        upper_triangular_diag_size = int(self.max_num_nodes * (self.max_num_nodes + 1) / 2)
        self.max_num_edges = int(self.max_num_nodes * (self.max_num_nodes - 1) / 2)
        self.num_edge_features = hparams["num_edge_features"]

        input_size = upper_triangular_diag_size + self.max_num_nodes * self.num_node_features + self.max_num_edges * self.num_edge_features

        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        adj_triu_mat, node_mat, edge_mat = x


        softmax = torch.nn.Softmax(dim=2)

        node_mat = softmax(node_mat)
        edge_mat = softmax(edge_mat)
        adj_triu_mat = F.sigmoid(adj_triu_mat)

        input = torch.cat(
            [
                torch.flatten(adj_triu_mat, start_dim=1),
                torch.flatten(node_mat, start_dim=1),
                torch.flatten(edge_mat, start_dim=1)
            ],
            dim=1,
        )

        return self.model(input)