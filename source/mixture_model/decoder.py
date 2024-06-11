import itertools
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


class MixtureModelDecoder(nn.Module):

    def __init__(self, hparams: Dict[str, Any]) -> None:
        super().__init__()

        self.num_clusters = hparams["num_clusters"]
        z_dim = hparams["z_dim"]
        num_atom_types = hparams["num_atom_types"]
        num_bond_types = hparams["num_bond_types"] + 1  # +1 for non-existent bonds

        # model parameters
        self.eta = nn.Parameter(torch.zeros(1, self.num_clusters))
        self.cluster_means = nn.Parameter(torch.randn(1, self.num_clusters, z_dim))
        self.cluster_log_sigmas = nn.Parameter(torch.ones(1, self.num_clusters, z_dim))

        atom_type_mlp_hidden_dim = hparams["atom_type_mlp_hidden_dim"]
        self.atom_classifier = nn.Sequential(
            nn.Linear(z_dim, atom_type_mlp_hidden_dim),
            nn.BatchNorm1d(atom_type_mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(atom_type_mlp_hidden_dim, num_atom_types)
        )
        self.bond_matrix = nn.Parameter(torch.randn(1, num_bond_types, z_dim, z_dim))
        
        self.gumbel_softmax_temperature = 1.0

        # create edge indices of fully connected graph for all graph sizes up to 50
        self.edge_indices = []
        for N in range(1, 50):
            self.edge_indices.append(
                torch.tensor(list(itertools.combinations(range(N), 2)))
            )

    def set_gumbel_softmax_temperature(self, temperature: float):
        self.gumbel_softmax_temperature = temperature
    
    def get_pi(self) -> torch.Tensor:
        return F.softmax(self.eta, dim=1)
    
    def sample_c(self, pi: torch.Tensor, num_atoms: torch.Tensor) -> torch.Tensor:
        # sample clusters using the gumbel-softmax reparameterization
        log_pi = torch.log(pi)
        log_pi = log_pi.expand(num_atoms.shape[0], log_pi.shape[1])
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
        edge_type_logits = (z_pairs[:, 0].permute(dims=(0, 1, 3, 2)) @ W @ z_pairs[:, 1]).squeeze()
        edge_types = F.softmax(edge_type_logits, dim=1).squeeze()

        edge_index = edge_index.t().contiguous()

        if self.training:
            # we don't need batch info during training
            # this saves us time and memory
            return Data(x=atom_types, edge_index=edge_index, edge_attr=edge_types)
        else:
            batch = torch.repeat_interleave(torch.arange(len(num_atoms), device=device), num_atoms)
            return Data(x=atom_types, edge_index=edge_index, edge_attr=edge_types, batch=batch)

    def forward(self, num_atoms: torch.Tensor) -> Data:
        pi = self.get_pi()
        c = self.sample_c(pi=pi, num_atoms=num_atoms)
        z = self.sample_z(c=c)
        return self.decode_z(z, num_atoms)

    def sample(self, num_atoms: torch.Tensor) -> Data:
        """
        Sample new molecules
        """
        return self.forward(num_atoms)