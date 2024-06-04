from __future__ import annotations
from typing import Tuple, Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse, remove_self_loops, to_undirected

import pandas as pd

from .encoder import Encoder
from .decoder import Decoder


class GraphVAE(nn.Module):

    def __init__(self, hparams: Dict[str, Any], prop_norm_df: Optional[pd.DataFrame] = None) -> None:
        super().__init__()

        self.encoder = Encoder(hparams=hparams)
        self.decoder = Decoder(hparams=hparams)
        self.latent_dim = hparams["latent_dim"]
        self.max_num_nodes = hparams["max_num_nodes"]
        self.num_node_features = hparams["num_node_features"]
        self.num_edge_features = hparams["num_edge_features"]
        self.property_z_size = hparams.get("property_latent_dim", self.latent_dim)
        dropout_p = hparams.get("property_model_dropout", 0)

        self.properties = hparams["properties"]
        self.num_properties = len(self.properties)
        property_predictor_hidden_dim = hparams.get("prop_net_hidden_dim", 67)
        if self.num_properties > 0:
            self.property_predictor = nn.Sequential(
                nn.Linear(self.property_z_size, property_predictor_hidden_dim),
                nn.BatchNorm1d(property_predictor_hidden_dim),
                nn.PReLU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(property_predictor_hidden_dim, property_predictor_hidden_dim),
                nn.BatchNorm1d(property_predictor_hidden_dim),
                nn.PReLU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(property_predictor_hidden_dim, self.num_properties * 2),
            )

        rows, cols = torch.triu_indices(self.max_num_nodes, self.max_num_nodes)
        self.diag_triu_mask = rows == cols

        self.edge_triu_rows, self.edge_triu_cols = torch.triu_indices(self.max_num_nodes, self.max_num_nodes, offset=1)

        # load normalization data        
        if prop_norm_df is not None:
            prop_norm_data = torch.tensor(prop_norm_df[self.properties].values, dtype=torch.float32)
            prop_mean = prop_norm_data[0]
            prop_std = prop_norm_data[1]
        else:
            # Default values; assuming that the norm data is loaded from a checkpoint later
            prop_mean = torch.tensor([0.0] * self.num_properties, dtype=torch.float32)
            prop_std = torch.tensor([1.0] * self.num_properties, dtype=torch.float32)

        self.register_buffer('prop_mean', prop_mean)
        self.register_buffer('prop_std', prop_std)


    @staticmethod
    def from_pretrained(checkpoint_path: str) -> GraphVAE:
        checkpoint = torch.load(checkpoint_path)
        graph_vae_model = GraphVAE(hparams=checkpoint["hparams"])
        graph_vae_model.load_state_dict(checkpoint['model_state_dict'])
        return graph_vae_model
    
    def normalize_properties(self, y):
        return (y - self.prop_mean) / self.prop_std

    def denormalize_properties(self, y):
        return y * self.prop_std + self.prop_mean
    
    def denormalize_properties_std(self, y_std):
        return y_std * self.prop_std

    def _sample_with_reparameterization(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        std_norm = torch.randn_like(mu)
        return std_norm * sigma + mu
    
    def z_to_property_z(self, z):
        """
        Return the portion of the latent space that is used for property prediction.
        """
        return z[:, :self.property_z_size]

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_sigma = self.encoder(data)

        log_sigma = torch.clamp(log_sigma, -30.0, 20.0)
        sigma = torch.exp(log_sigma)

        z = self._sample_with_reparameterization(mu=mu, sigma=sigma)
        x = self.decoder(z)

        if self.num_properties > 0:
            y = self.property_predictor(self.z_to_property_z(z))
            y_mu = y[:, :self.num_properties]
            log_sigma = y[:, self.num_properties:]
            y_sigma = torch.exp(torch.clamp(log_sigma, -20, 30))
        else:
            y_mu = None
            y_sigma = None

        return x, mu, sigma, y_mu, y_sigma
    
    @staticmethod
    def kl_divergence(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma_squared = sigma * sigma
        log_sigma_squared = torch.log(sigma_squared)
        mu_squared = mu * mu
        # technically this is the correct KL-divergence
        # but in practice it is way too high
        kl_div_sample = 0.5 * torch.sum(sigma_squared + mu_squared - log_sigma_squared - 1, dim=1)
        #kl_div_sample = 0.5 * torch.mean(sigma_squared + mu_squared - log_sigma_squared - 1, dim=1)
        # average over the batch
        return torch.mean(kl_div_sample)
    
    @staticmethod
    def pairwise_kl_divergence(tensor_p, tensor_q):
        # Expand tensors for broadcasting
        tensor_p_expanded = tensor_p.unsqueeze(2)
        tensor_q_expanded = tensor_q.unsqueeze(1)

        kl_div = F.kl_div(tensor_p_expanded.log(), tensor_q_expanded, reduction='none')
        
        # Sum over the last dimension to get the final KL divergence values
        kl_div_summed = kl_div.sum(dim=-1)

        return kl_div_summed

    
    def reconstruction_loss(
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
        if not isinstance(x, Batch):
            x = Batch.from_data_list(data_list=[x])

        mu, log_sigma = self.encoder(x)

        log_sigma = torch.clamp(log_sigma, -30.0, 20.0)
        sigma = torch.exp(log_sigma)

        z = self._sample_with_reparameterization(mu=mu, sigma=sigma)
        return z
    
    def encode_mean(self, x: Data):
        if not isinstance(x, Batch):
            x = Batch.from_data_list(data_list=[x])
        mu, _ = self.encoder(x)
        return mu

    def sample(self, num_samples: int, device: str):
        z = torch.randn((num_samples, self.latent_dim), device=device)
        x = self.decoder(z)
        return z, x


    def predict_properties(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.num_properties == 0:
            raise ValueError("Model has not been trained with property prediction")
        y = self.property_predictor(self.z_to_property_z(z))
        y_mu = y[:, :self.num_properties]
        log_sigma = y[:, self.num_properties:]
        y_sigma = torch.exp(torch.clamp(log_sigma, -20, 30))
        return y_mu, y_sigma


    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


    def output_to_graph(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], stochastic: bool) -> Data:
        """
        Convert one batch element of the decoder output, which consists of the adjacency matrix,
        the node attributes matrix and the edge attribute matrix into a PyTorch Geometric graph object.
        """

        pred_adj_triu_mat = x[0][0]
        pred_node_mat = x[1][0]
        pred_edge_triu_mat = x[2][0]

        device = pred_adj_triu_mat.device
        n = self.max_num_nodes

        if stochastic:
            softmax = torch.nn.Softmax(dim=1)
            normalized_pred_edge_triu_mat = softmax(pred_edge_triu_mat)
            edge_triu_mat = torch.multinomial(normalized_pred_edge_triu_mat, num_samples=1)[:, 0].float()
        else:
            edge_triu_mat = pred_edge_triu_mat.argmax(dim=1).float()

        edge_mat = torch.zeros(n, n, device=device)
        edge_mat[torch.ones(n, n).triu(diagonal=1) == 1] = edge_triu_mat
        edge_mat = edge_mat + 1  # add one so we can so that 0 indicates no node instead of hydrogen

        # convert predicted upper triagular matrix into symmetric edge index and
        if stochastic:
            adj_triu_mat = torch.bernoulli(F.sigmoid(pred_adj_triu_mat))
        else:
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
        if stochastic:
            softmax = torch.nn.Softmax(dim=1)
            normalized_pred_node_mat = softmax(pred_node_mat[node_mask])
            node_feature_sample = torch.multinomial(normalized_pred_node_mat, num_samples=1)[:, 0]
        else:
            node_feature_sample = pred_node_mat[node_mask].argmax(dim=1)
        x = F.one_hot(node_feature_sample, num_classes=self.num_node_features)

        edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr.float())
        return Data(x=x.float(), edge_index=edge_index, edge_attr=edge_attr)
    