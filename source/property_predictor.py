from typing import Dict, Any

import torch
import torch.nn as nn

from torch_geometric.data import Data

from graph_vae.encoder import Encoder


class PropertyPredictor(nn.Module):

    def __init__(self, hparams: Dict[str, Any]) -> None:
        super().__init__()
        self.graph_encoder = Encoder(hparams=hparams)
        property_count = len(hparams["properties"])
        dim = hparams["latent_dim"] * 2
        self.fc = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.PReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.PReLU(),
            nn.Linear(dim, property_count)
        )

    def forward(self, x: Data):
        z = self.graph_encoder(x)
        # combine tuple (mu, log_sigma) into single latent
        z = torch.cat(list(z), dim=1)
        return self.fc(z)

def load_property_predictor_model() -> PropertyPredictor:
    # TODO: remove this function; just added for the prof demo
    hparams = {
        "max_num_nodes": 9,
        "num_node_features": 4,
        "num_edge_features": 4,
        "latent_dim": 64,
        "include_hydrogen": False,
        "properties": ["homo", "lumo"],
    }
    model = PropertyPredictor(hparams=hparams)
    ckpt_file = "./checkpoints/property_predictor.pt"
    model.load_state_dict(torch.load(ckpt_file)["model_state_dict"])
    return model

    