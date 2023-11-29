from typing import List, Union

import torch
from torch.utils.data import random_split
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, HeteroData


class SelectQM9TargetProperties(BaseTransform):
    """
    Ensure that only the specified properties are included in the target y.
    """

    def __init__(self, properties: List[str]):
        property_names = [
            "mu", "alpha", "homo", "lumo", "gap", "r2",
            "zpve", "U0", "U", "H", "G", "Cv", "U0_atom",
            "U_atom", "H_atom", "G_atom", "A", "B", "C"
        ]
        property_name_to_index = {
            name: index for index, name in enumerate(property_names)
        }
        self.indices = [property_name_to_index[name] for name in properties]

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        data.y = data.y[:, self.indices]
        return data


class SelectQM9NodeFeatures(BaseTransform):

    def __init__(self, features: List[str]):
        feature_name_to_index_map = {
            "atom_type": list(range(5)),  # one hot encoding
            "atomic_number": [5],
            "aromatic": [6],  # 1 or 0 (true or false)
            "hybridization": list(range(3)),  # one hot encoding (sp,sp2,sp3)
            "num_hs": [10]  # number of hydrogen atom connected to this atom
        }
        self.indices = [i for name in features for i in feature_name_to_index_map[name]]

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        data.x = data.x[:, self.indices]
        return data

def create_qm9_data_split(dataset):
    generator = torch.manual_seed(420)
    return random_split(dataset=dataset, lengths=[0.8, 0.1, 0.1], generator=generator)