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

        self.num_properties = len(hparams["properties"])
        if self.num_properties > 0:
            self.properties_predictor = nn.Sequential(
                nn.Linear(self.latent_dim, 67),
                nn.BatchNorm1d(67),
                nn.ReLU(),
                nn.Linear(67, 67),
                nn.BatchNorm1d(67),
                nn.ReLU(),
                nn.Linear(67, self.num_properties),
            )


        rows, cols = torch.triu_indices(self.max_num_nodes, self.max_num_nodes)
        self.diag_triu_mask = rows == cols

        self.edge_triu_rows, self.edge_triu_cols = torch.triu_indices(self.max_num_nodes, self.max_num_nodes, offset=1)

    @staticmethod
    def from_pretrained(checkpoint_path: str) -> GraphVAE:
        checkpoint = torch.load(checkpoint_path)
        graph_vae_model = GraphVAE(hparams=checkpoint["hparams"])
        graph_vae_model.load_state_dict(checkpoint['model_state_dict'])
        return graph_vae_model

    def _sample_with_reparameterization(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        std_norm = torch.randn_like(mu)
        return std_norm * sigma + mu

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(data)

        log_var = torch.clamp(log_var, -30.0, 20.0)
        sigma = torch.exp(log_var / 2)

        z = self._sample_with_reparameterization(mu=mu, sigma=sigma)
        x = self.decoder(z)

        if self.num_properties > 0:
            y = self.properties_predictor(z)
        else:
            y = None

        return x, mu, sigma, y
    
    @staticmethod
    def kl_divergence(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma_squared = sigma * sigma
        log_sigma_squared = torch.log(sigma_squared)
        mu_squared = mu * mu
        # technically this is the correct KL-divergence
        # but in practice it is way too high
        # kl_div_sample = 0.5 * torch.sum(sigma_squared + mu_squared - log_sigma_squared - 1, dim=1)
        kl_div_sample = 0.5 * torch.mean(sigma_squared + mu_squared - log_sigma_squared - 1, dim=1)
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

        # graph matching
        # input_node_distribution = torch.nn.Softmax(dim=2)(input_node_mat)
        # target_node_distribution = torch.nn.Softmax(dim=2)(target_node_mat)

        # kl_div_mat = self.pairwise_kl_divergence(tensor_p=input_node_distribution, tensor_q=target_node_distribution)

        # assignment_matrices = []
        # for S in kl_div_mat:
        #     # Convert to cost matrix if necessary
        #     cost_matrix = S.max() - S
        #     from scipy.optimize import linear_sum_assignment
        #     # Apply the Hungarian algorithm
        #     row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())

        #     # Create the assignment matrix X
        #     X = torch.zeros_like(cost_matrix, device=kl_div_mat.device)
        #     X[row_ind, col_ind] = 1
        #     # Store the result
        #     assignment_matrices.append(X.unsqueeze(0))

        # assigment_matrix = torch.cat(assignment_matrices)
        # device = assigment_matrix.device

        # # permute adjacency matrix
        # n = self.max_num_nodes
        # input_adj_mat = torch.zeros(input_adj_triu_mat.shape[0], n, n, device=device)
        # triu_mask_adj = torch.ones(n, n).triu() == 1
        # input_adj_mat[:, triu_mask_adj] = input_adj_triu_mat
        # input_adj_mat = torch.bmm(assigment_matrix, torch.bmm(input_adj_mat, assigment_matrix.transpose(1, 2)))
        
        # input_adj_triu_mat = input_adj_mat[:, triu_mask_adj]

        # # permute node features
        # input_node_mat = torch.bmm(assigment_matrix.transpose(1, 2), input_node_mat)

        # # permute edge features
        # edge_mat = torch.zeros(input_edge_mat.shape[0], n, n, 4, device=device)
        # triu_mask_edge = torch.ones(n, n).triu(diagonal=1) == 1
        # edge_mat[:, triu_mask_edge] = input_edge_mat

        # for i in range(edge_mat.shape[3]):
        #     edge_mat[:, :, :, i] =  torch.bmm(assigment_matrix, torch.bmm(edge_mat[:, :, :, i], assigment_matrix.transpose(1, 2)))

        # input_edge_mat = edge_mat[:, triu_mask_edge]

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
        mu, log_var = self.encoder(x)

        log_var = torch.clamp(log_var, -30.0, 20.0)
        sigma = torch.exp(log_var / 2)

        z = self._sample_with_reparameterization(mu=mu, sigma=sigma)
        return z
        

    def sample(self, num_samples: int, device: str):
        z = torch.randn((num_samples, self.latent_dim), device=device)
        x = self.decoder(z)
        return z, x
    

    def output_to_graph(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], stochastic: bool) -> Data:
        # TODO: handle batches
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

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    