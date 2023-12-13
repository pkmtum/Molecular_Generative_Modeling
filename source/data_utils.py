from typing import List, Union, Tuple
import os
import math

import torch
from torch.utils.data import random_split
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, HeteroData, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, add_self_loops
from torch.utils.tensorboard import SummaryWriter

from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np

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
    
class AddAdjacencyMatrix(BaseTransform):
    """ 
    Create the upper triangular part of the adjacency matrix from the edge_index.
    The matrix is padded with zeors to a shape of (max_num_nodes, max_num_nodes).
    The binary diagonal elements indicated the presence of a node.
    The result is stored in the adj_triu_mat attribute.
    """

    def __init__(self, max_num_nodes: int) -> None:
        self.max_num_nodes = max_num_nodes
        self.triu_mask = torch.ones(max_num_nodes, max_num_nodes).triu() == 1

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        edge_index_with_loops, _ = add_self_loops(edge_index=data.edge_index)
        adj_mat = to_dense_adj(edge_index=edge_index_with_loops, max_num_nodes=self.max_num_nodes)
        data.adj_triu_mat = adj_mat[:, self.triu_mask]
        return data
    
class AddNodeAttributeMatrix(BaseTransform):
    """
    Add the node attribute matrix. 
    """
    def __init__(self, max_num_nodes: int) -> None:
        self.max_num_nodes = max_num_nodes

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        num_nodes_to_pad = self.max_num_nodes - data.x.shape[0]

        # pad with hydrogen
        padding_value = [1, 0, 0, 0, 0]

        padding_tensor = torch.tensor([padding_value] * num_nodes_to_pad)
        data.node_mat = torch.cat((data.x, padding_tensor), dim=0).unsqueeze(0)

        return data
    
def remove_nodes(data, nodes_to_remove):
    # Convert to set for faster lookup
    nodes_to_remove = set(nodes_to_remove)

    # Filter out edges connected to nodes to be removed
    mask = ~torch.tensor([data.edge_index[0][i].item() in nodes_to_remove or 
                          data.edge_index[1][i].item() in nodes_to_remove 
                          for i in range(data.edge_index.size(1))])

    new_edge_index = data.edge_index[:, mask]

    # Create a mapping for new node indices
    node_mapping = {}
    new_index = 0
    for old_index in range(data.num_nodes):
        if old_index not in nodes_to_remove:
            node_mapping[old_index] = new_index
            new_index += 1

    # Update edge indices to reflect new node indices
    for i in range(new_edge_index.size(1)):
        new_edge_index[0][i] = node_mapping[new_edge_index[0][i].item()]
        new_edge_index[1][i] = node_mapping[new_edge_index[1][i].item()]

    # Update node features
    new_x = data.x[torch.tensor([i for i in range(data.num_nodes) if i not in nodes_to_remove])]

    # Create new data object
    new_data = Data(x=new_x, edge_index=new_edge_index)

    return new_data

def create_qm9_data_split(dataset) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create a training, validation and test set from the full QM9 dataset.
    """
    generator = torch.manual_seed(420)
    return random_split(dataset=dataset, lengths=[0.8, 0.1, 0.1], generator=generator)


def smiles_to_image(smiles: str) -> torch.tensor:
    mol = Chem.MolFromSmiles(smiles)
    image = Draw.MolToImage(mol)
    image = np.array(image)
    # Convert to CHW format
    tensor = torch.tensor(np.transpose(image, (2, 0, 1)))
    # Add batch dimension
    return tensor.unsqueeze(0)

def molecule_graph_data_to_image(data: Data, validate: bool = True) -> torch.tensor:
    # create empty molecule
    mol = Chem.RWMol()

    class_index_to_atomic_number = {
        0: 1, 1: 6, 2: 7, 3: 8, 4: 9
    }
    # Add atoms
    for atom_features in data.x:
        # convert the one-hot encoded atom class to the atomic number
        class_index = torch.argmax(atom_features[:5]).item()
        atomic_number = class_index_to_atomic_number[class_index]
        atom = Chem.Atom(int(atomic_number))
        mol.AddAtom(atom)  

    # Create set of undirected bonds
    undirected_bonds = set()
    for edge_indices, edge_feature in zip(data.edge_index.t(), data.edge_attr):
        start_atom, end_atom = edge_indices
        bond_type_index = torch.argmax(edge_feature).item()
        bond = tuple(sorted((start_atom.item(), end_atom.item())) + [bond_type_index])
        undirected_bonds.add(bond)

    bond_type_map = {
        0: Chem.BondType.SINGLE,
        1: Chem.BondType.DOUBLE,
        2: Chem.BondType.TRIPLE,
        3: Chem.BondType.AROMATIC
    }
    # Add bonds
    for start_atom, end_atom, bond_type_index in undirected_bonds:
        mol.AddBond(int(start_atom), int(end_atom), bond_type_map[bond_type_index])

    # Check if the molecule is chemically valid
    # try:
    #     Chem.SanitizeMol(mol)
    # except Exception as e:
    #     print(f"Chemically invalid molecule! Reason: {e}")
    
    # Convert to a standard RDKit mol object
    mol = mol.GetMol()

    # Remove hydrogen atoms for visualization
    #mol = Chem.RemoveHs(mol, sanitize=False)

    image = Draw.MolToImage(mol)
    image = np.array(image)
    # Convert to CHW format
    tensor = torch.tensor(np.transpose(image, (2, 0, 1)))
    # Add batch dimension
    return tensor.unsqueeze(0)


def create_tensorboard_writer(experiment_name: str, log_dir_root: str = "./tb_logs"):
    logdir = os.path.join(log_dir_root, experiment_name)
    os.makedirs(logdir, exist_ok=True)
    experiment_index = len(os.listdir(logdir))
    return SummaryWriter(os.path.join(logdir, str(experiment_index).zfill(3)))


def create_validation_subset_loaders(validation_dataset, subset_count, batch_size):
    """ Create random subsets of the validation set for fast validation. """
    validation_subsets = []
    generator = torch.manual_seed(420)
    validation_indices = torch.randperm(len(validation_dataset), generator=generator).tolist()
    subset_size = math.ceil(len(validation_dataset) / subset_count)
    for i in range(subset_count):
        start_index = subset_size * i
        end_index = min(subset_size * (i + 1), len(validation_dataset))
        val_subset = torch.utils.data.Subset(validation_dataset, validation_indices[start_index:end_index])
        validation_subsets.append(DataLoader(val_subset, batch_size=batch_size, shuffle=False))
    return validation_subsets