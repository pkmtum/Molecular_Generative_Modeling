from __future__ import annotations
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm


from .encoder import MixtureModelEncoder
from .decoder import MixtureModelDecoder


class MixtureModel(nn.Module):

    def __init__(self,  hparams: Dict[str, Any]) -> None:
        super().__init__()

        self.encoder = MixtureModelEncoder(hparams=hparams)
        self.decoder = MixtureModelDecoder(hparams=hparams)

        self.uniform_cluster_probs = hparams["uniform_cluster_probs"]

    @staticmethod
    def from_pretrained(checkpoint_path: str) -> MixtureModel:
        checkpoint = torch.load(checkpoint_path)
        graph_vae_model = MixtureModel(hparams=checkpoint["hparams"])
        graph_vae_model.load_state_dict(checkpoint['model_state_dict'])
        return graph_vae_model
    
    def encode(self, x: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        z_mu = encoded[0]
        z_sigma = encoded[1]
        z = torch.randn_like(z_mu) * z_sigma + z_mu
        if self.uniform_cluster_probs:
            return z

        eta_mu = encoded[2]
        eta_sigma = encoded[3]
        eta = torch.randn_like(eta_mu) * eta_sigma + eta_mu
        return z, eta
        