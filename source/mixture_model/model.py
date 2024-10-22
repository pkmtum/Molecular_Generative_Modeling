from __future__ import annotations
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .encoder import MixtureModelEncoder
from .decoder import MixtureModelDecoder


class MixtureModel(nn.Module):

    def __init__(self,  hparams: Dict[str, Any]) -> None:
        super().__init__()

        self.encoder = MixtureModelEncoder(hparams=hparams)
        self.decoder = MixtureModelDecoder(hparams=hparams)

    @staticmethod
    def from_pretrained(checkpoint_path: str) -> MixtureModel:
        checkpoint = torch.load(checkpoint_path)
        graph_vae_model = MixtureModel(hparams=checkpoint["hparams"])
        graph_vae_model.load_state_dict(checkpoint['model_state_dict'])
        return graph_vae_model
    
    def encode(self, x: Data) -> torch.Tensor:
        encoded = self.encoder(x)
        z_mu = encoded[0]
        z_sigma = encoded[1]
        z = torch.randn_like(z_mu) * z_sigma + z_mu
        return z
