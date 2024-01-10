from __future__ import annotations
from typing import Tuple, Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, remove_self_loops

from .encoder import Encoder
from .decoder import Decoder


class GraphVAE(nn.Module):

    def __init__(self, hparams: Dict[str, Any]) -> None:
        super().__init__()

        self.encoder = Encoder(hparams=hparams)
        self.decoder = Decoder(hparams=hparams)
        self.latent_dim = hparams["latent_dim"]
        self.max_num_nodes = hparams["max_num_nodes"]
        self.num_node_features = hparams["num_node_features"]
        self.num_edge_features = hparams["num_edge_features"]
        self.kl_weight = hparams["kl_weight"]

        rows, cols = torch.triu_indices(self.max_num_nodes, self.max_num_nodes)
        self.diag_triu_mask = rows == cols

        self.edge_triu_rows, self.edge_triu_cols = torch.triu_indices(self.max_num_nodes, self.max_num_nodes, offset=1)

    @staticmethod
    def from_pretrained(checkpoint_path: str) -> GraphVAE:
        checkpoint = torch.load(checkpoint_path)
        graph_vae_model = GraphVAE(hparams=checkpoint["hparams"])
        graph_vae_model.load_state_dict(checkpoint['model_state_dict'])
        return graph_vae_model

    def _sample_with_reparameterization(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(log_sigma)
        std_norm = torch.randn_like(mu)
        return std_norm * sigma + mu

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_sigma = self.encoder(data)
        z = self._sample_with_reparameterization(mu=mu, log_sigma=log_sigma)
        x = self.decoder(z)
        return x
    
    def _kl_divergence(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        log_sigma_squared = log_sigma + log_sigma
        sigma_squared = torch.exp(log_sigma_squared)
        mu_squared = mu * mu
        kl_div_sample = 0.5 * torch.sum(sigma_squared + mu_squared - log_sigma_squared - 1, dim=1)
        # average over the batch
        return torch.mean(kl_div_sample)
    
    def _reconstruction_loss(
        self, 
        input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
        target: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        input_adj_triu_mat, input_node_mat, input_edge_mat = input
        target_adj_triu_mat, target_node_mat, target_edge_mat = target

        # average loss over nodes and edges separately
        input_adj_triu_mat_diag = input_adj_triu_mat[:, self.diag_triu_mask]
        input_adj_triu_mat_off_diag = input_adj_triu_mat[:, ~self.diag_triu_mask]
        target_adj_triu_mat_diag = target_adj_triu_mat[:, self.diag_triu_mask]
        target_adj_triu_mat_off_diag = target_adj_triu_mat[:, ~self.diag_triu_mask]
        adjacency_loss = (
            F.binary_cross_entropy_with_logits(input=input_adj_triu_mat_diag, target=target_adj_triu_mat_diag)
            + F.binary_cross_entropy_with_logits(input=input_adj_triu_mat_off_diag, target=target_adj_triu_mat_off_diag)
        )

        # compute the node feature loss only for nodes that exist in the input graph
        node_mat_logits = input_node_mat.view(-1, input_node_mat.size(-1))
        node_targets = target_node_mat.argmax(dim=2).view(-1)
        per_node_feature_loss = F.cross_entropy(input=node_mat_logits, target=node_targets, reduction="none")
        node_mask = target_adj_triu_mat_diag.view(-1)
        node_feature_loss = (per_node_feature_loss * node_mask).sum() / node_mask.sum()

        # Compute edge feature loss only for edges that are connected to nodes existing in the input graph
        edge_mat_logits = input_edge_mat.view(-1, input_edge_mat.size(-1))
        edge_targets = target_edge_mat.argmax(dim=2).view(-1)
        per_edge_feature_loss = F.cross_entropy(input=edge_mat_logits, target=edge_targets, reduction="none")
        edge_mask = (
            target_adj_triu_mat_diag[:, self.edge_triu_rows].int() & target_adj_triu_mat_diag[:, self.edge_triu_cols].int()
        ).view(-1)
        edge_feature_loss = (per_edge_feature_loss * edge_mask).sum() / edge_mask.sum()
        
        return adjacency_loss + node_feature_loss + edge_feature_loss
    

    def encode(self, x: Data):
        mu, log_sigma = self.encoder(x)
        z = self._sample_with_reparameterization(mu=mu, log_sigma=log_sigma)
        return z
    
    
    def elbo(self, x: Data):
        mu, log_sigma = self.encoder(x)
        z = self._sample_with_reparameterization(mu=mu, log_sigma=log_sigma)

        x_recon = self.decoder(z)
        x_target = (x.adj_triu_mat, x.node_mat, x.edge_triu_mat)
        recon_loss = self._reconstruction_loss(input=x_recon, target=x_target)
        kl_div = self._kl_divergence(mu=mu, log_sigma=log_sigma)
        elbo = -(recon_loss + kl_div * self.kl_weight)
        return elbo, recon_loss
    

    def sample(self, num_samples: int, device: str):
        z = torch.randn((num_samples, self.latent_dim), device=device)
        x = self.decoder(z)
        return z, x
    

    def output_to_graph(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Data:
        # TODO: handle batches
        pred_adj_triu_mat = x[0][0]
        pred_node_mat = x[1][0]
        pred_edge_triu_mat = x[2][0]

        device = pred_adj_triu_mat.device
        n = self.max_num_nodes

        edge_triu_mat = pred_edge_triu_mat.argmax(dim=1).float()
        edge_mat = torch.zeros(n, n, device=device)
        edge_mat[torch.ones(n, n).triu(diagonal=1) == 1] = edge_triu_mat
        edge_mat = edge_mat + 1  # add one so we can so that 0 indicates no node instead of hydrogen

        # convert predicted upper triagular matrix into symmetric edge index and
        adj_triu_mat = torch.where(F.sigmoid(pred_adj_triu_mat) > 0.5, 1.0, 0.0)
        adj_mat = torch.zeros(n, n, device=device)
        adj_mat[torch.ones(n, n).triu() == 1] = adj_triu_mat
        diagonal = adj_mat.diagonal()

        # combine the adjacency matrix with edge features
        edge_mat *= adj_mat

        node_mask = diagonal == 1
        edge_index, edge_attr = dense_to_sparse(adj=edge_mat.unsqueeze(0), mask=node_mask.unsqueeze(0))

        edge_attr = F.one_hot((edge_attr - 1).long(), num_classes=self.num_edge_features)

        # Adjust indices in edge_index to account for removed nodes
        # Create a mapping from old indices to new indices
        old_to_new_indices = torch.cumsum(node_mask, 0) - 1
        old_to_new_indices[~node_mask] = -1
        new_edge_index = old_to_new_indices[edge_index]

        # Remove edges that contain removed nodes
        valid_edges = (new_edge_index >= 0).all(dim=0)
        new_edge_index = new_edge_index[:, valid_edges]
        new_edge_attr = edge_attr[valid_edges, :]

        edge_index, edge_attr = remove_self_loops(edge_index=new_edge_index, edge_attr=new_edge_attr)

        # convert node feature logits into one-hot vector
        x = F.one_hot(pred_node_mat[node_mask].argmax(dim=1), num_classes=self.num_node_features)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    