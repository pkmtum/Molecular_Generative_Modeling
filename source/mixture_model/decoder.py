import itertools
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from .common import ResidualBlock


class MixtureModelDecoder(nn.Module):

    def __init__(self, hparams: Dict[str, Any]) -> None:
        super().__init__()

        self.eta_dim = hparams["eta_dim"]
        num_clusters = hparams["num_clusters"]
        self.num_clusters = num_clusters
        z_dim = hparams["z_dim"]
        num_atom_types = hparams["num_atom_types"]
        num_bond_types = hparams["num_bond_types"] + 1  # +1 for non-existent bonds
        dropout_p = hparams["dropout"]

        # model parameters
        self.eta_mu = nn.Parameter(torch.zeros(1, self.eta_dim))
        self.eta_log_sigma = nn.Parameter(torch.zeros(1, self.eta_dim))
        cluster_mlp_hidden_dim = hparams["cluster_mlp_hidden_dim"]
        if cluster_mlp_hidden_dim > 0:
            self.cluster_mlp = ResidualBlock(
                in_features=self.eta_dim,
                hidden_features=cluster_mlp_hidden_dim,
                out_features=self.num_clusters,
                dropout=dropout_p,
            )
        else:
            self.cluster_mlp = nn.Identity()

        self.cluster_means = nn.Parameter(torch.randn(1, self.num_clusters, z_dim))
        self.cluster_log_sigmas = nn.Parameter(torch.ones(1, self.num_clusters, z_dim) * -2.0)
        atom_type_mlp_hidden_dim = hparams["atom_type_mlp_hidden_dim"]
        self.atom_classifier = ResidualBlock(
            in_features=z_dim,
            hidden_features=atom_type_mlp_hidden_dim,
            out_features=num_atom_types,
            dropout=dropout_p
        )
        bond_type_mlp_hidden_dim = hparams["bond_type_mlp_hidden_dim"]
        if bond_type_mlp_hidden_dim > 0:
            self.bond_matrix = nn.Parameter(torch.randn(1, num_bond_types, z_dim, z_dim))
            self.bond_type_mlp = ResidualBlock(
                in_features=num_bond_types,
                hidden_features=bond_type_mlp_hidden_dim,
                out_features=num_bond_types,
                dropout=dropout_p
            )
        else:
            self.bond_matrix = nn.Parameter(torch.randn(1, num_bond_types, z_dim, z_dim))
            self.bond_type_mlp = nn.Identity()
        
        self.gumbel_softmax_temperature = 1.0

        # create edge indices of fully connected graph for all graph sizes up to 50
        self.edge_indices = []
        for N in range(1, 50):
            self.edge_indices.append(
                torch.tensor(list(itertools.combinations(range(N), 2)))
            )

    def set_gumbel_softmax_temperature(self, temperature: float):
        self.gumbel_softmax_temperature = temperature

    def sample_eta(self, num_atoms: torch.Tensor, device: str) -> torch.Tensor:
        randn = torch.randn(size=(num_atoms.size(0), self.eta_dim), device=device)
        eta_sigma = torch.exp(torch.clamp(self.eta_log_sigma, -20, 30))
        eta = randn * eta_sigma + self.eta_mu
        return eta

    def decode_eta(self, eta: torch.Tensor) -> torch.Tensor:
        pi = F.softmax(self.cluster_mlp(eta), dim=1)
        return pi
    
    def sample_c(self, pi: torch.Tensor, num_atoms: torch.Tensor) -> torch.Tensor:
        # sample clusters using the gumbel-softmax reparameterization
        log_pi = torch.log(pi)
        c = F.gumbel_softmax(
            logits=torch.repeat_interleave(log_pi, num_atoms, dim=0),
            tau=self.gumbel_softmax_temperature,
            hard=True
        ).unsqueeze(-1)
        return c
    
    def sample_z(self, c: torch.Tensor) -> torch.Tensor:
        mu = torch.sum(self.cluster_means * c, dim=1)
        log_sigma = torch.sum(self.cluster_log_sigmas * c, dim=1)
        sigma = torch.exp(torch.clamp(log_sigma, -20, 30))
        z = torch.randn_like(mu) * sigma + mu
        return z
    
    def decode_z(self, z: torch.Tensor, num_atoms: torch.Tensor) -> Data:
        atom_types = self.atom_classifier(z)

        device = z.device
        
        # create edge indices of fully connected graphs
        edge_index_list = []
        offsets = torch.cat([torch.tensor([0], device=device), torch.cumsum(num_atoms, dim=0)[:-1]])
        for i, N in enumerate(num_atoms):
            pairs = self.edge_indices[N - 1].to(device)
            pairs += offsets[i]
            edge_index_list.append(pairs)

        edge_index = torch.cat(edge_index_list, dim=0)

        # make bond matrix symmetric to ensure permuation invariance
        W = (self.bond_matrix + self.bond_matrix.transpose(2, 3)) * 0.5

        z_pairs = z[edge_index].unsqueeze(-2).unsqueeze(-1)
        edge_type_logits = self.bond_type_mlp((z_pairs[:, 0].permute(dims=(0, 1, 3, 2)) @ W @ z_pairs[:, 1]).squeeze())
        edge_types = F.softmax(edge_type_logits, dim=1).squeeze()

        edge_index = edge_index.t().contiguous()

        if self.training:
            # we don't need batch info during training
            return Data(x=atom_types, edge_index=edge_index, edge_attr=edge_types)
        else:
            batch = torch.repeat_interleave(torch.arange(len(num_atoms), device=device), num_atoms)
            return Data(x=atom_types, edge_index=edge_index, edge_attr=edge_types, batch=batch)

    def forward(self, eta: torch.Tensor, num_atoms: torch.Tensor) -> Data:
        pi = self.decode_eta(eta)
        c = self.sample_c(pi=pi, num_atoms=num_atoms)
        z = self.sample_z(c=c)
        return self.decode_z(z, num_atoms)

    def sample(self, num_atoms: torch.Tensor, device: str) -> Data:
        """
        Sample new molecules
        """
        eta = self.sample_eta(num_atoms=num_atoms, device=device)
        return self.forward(eta, num_atoms)