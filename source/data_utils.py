from typing import List, Union, Tuple, Optional
import os
import math
import shutil
import itertools

import torch
from torch.utils.data import random_split
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, HeteroData, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj, add_self_loops
from torch.utils.tensorboard import SummaryWriter

from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import pandas as pd
from tqdm import tqdm


# global constants
DATA_ROOT_DIR = "./data"
GRAPH_VAE_DATA_ROOT_DIR = os.path.join(DATA_ROOT_DIR, "graph_vae")
MIXTURE_VAE_DATA_ROOT_DIR = os.path.join(DATA_ROOT_DIR, "mixture_vae")
QM9_PROPERTIES = [
    "mu", "alpha", "homo", "lumo", "gap", "r2",
    "zpve", "U0", "U", "H", "G", "Cv", "U0_atom",
    "U_atom", "H_atom", "G_atom", "A", "B", "C"
]


class SelectQM9TargetProperties(BaseTransform):
    """
    Ensure that only the specified properties are included in the target y.
    """

    def __init__(self, properties: List[str]):
        property_names = QM9_PROPERTIES
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
    

class NormalizeQM9Properties(BaseTransform):

    def __init__(self, properties: Optional[List[str]], prop_norm_df: pd.DataFrame) -> None:
        self.properties = properties if properties is not None else QM9_PROPERTIES
        prop_norm_data = torch.tensor(prop_norm_df[properties].values, dtype=torch.float32)
        self.prop_mean = prop_norm_data[0]
        self.prop_std = prop_norm_data[1]

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        data.y = (data.y - self.prop_mean) / self.prop_std
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
    
class DropQM9Hydrogen(BaseTransform):
    """ Remove hydrogen atoms and all connected bond from the molecular graph. """

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        nodes_to_remove = torch.where(data.x.argmax(dim=1) == 0)[0]
        
        mask = torch.ones(data.num_nodes, dtype=torch.bool)
        mask[nodes_to_remove] = False

        data.x = data.x[mask]
        data.x = data.x[:, 1:]

        # remove edge indices and attributes
        edge_mask = torch.isin(data.edge_index, nodes_to_remove, invert=True).all(dim=0)
        data.edge_index = data.edge_index[:, edge_mask]
        data.edge_attr = data.edge_attr[edge_mask]

        # udpate index mapping
        index_mapping = torch.cumsum(mask, 0) - 1
        data.edge_index = index_mapping[data.edge_index]

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

        # padding with hydrogen
        padding_value = [1] + [0] * (data.x.shape[1] - 1)

        padding_tensor = torch.tensor([padding_value] * num_nodes_to_pad)
        data.node_mat = torch.cat((data.x, padding_tensor), dim=0).unsqueeze(0)

        return data
    
class AddEdgeAttributeMatrix(BaseTransform):
    """ 
    Create the upper triangular part of the edge attribute matrix.
    The matrix is padded with zeors to a shape of (max_num_nodes, max_num_nodes).
    The result is stored in the edge_mat attribute.
    """

    def __init__(self, max_num_nodes: int) -> None:
        self.max_num_nodes = max_num_nodes
        self.triu_mask = torch.ones(max_num_nodes, max_num_nodes).triu(diagonal=1) == 1

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        adj_mat = to_dense_adj(edge_index=data.edge_index, edge_attr=data.edge_attr, max_num_nodes=self.max_num_nodes)
        data.edge_triu_mat = adj_mat[:, self.triu_mask]
        return data
    
class RandomPermutation(BaseTransform):
    """ 
    Randomly permutes the adjacency matrix of a graphs.
    Also permutes the edge and node attribute matrices accordingly.
    """

    def __init__(self, max_num_nodes: int) -> None:
        self.max_num_nodes = max_num_nodes
        self.triu_mask_adj = torch.ones(max_num_nodes, max_num_nodes).triu() == 1
        self.triu_mask_edge = torch.ones(max_num_nodes, max_num_nodes).triu(diagonal=1) == 1

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        adj_triu_mat = data.adj_triu_mat
        node_mat = data.node_mat
        edge_triu_mat = data.edge_triu_mat

        batch_size = adj_triu_mat.shape[0]
        device = adj_triu_mat.device
        n = self.max_num_nodes

        permutations = torch.rand(batch_size, n).argsort(dim = 1)

        # construct full edge attribute matrix from upper triangular
        edge_mat = torch.zeros(batch_size, n, n, 4, device=device)
        edge_mat[:, self.triu_mask_edge] = edge_triu_mat
        edge_mat = edge_mat + edge_mat.transpose(1, 2)
        
        # construct full adjacency matrix from upper triangular
        adj_mat = torch.zeros(batch_size, n, n, device=device)
        adj_mat[:, self.triu_mask_adj == 1] = adj_triu_mat
        adj_mat = adj_mat + adj_mat.transpose(1, 2)
        adj_mat.diagonal(dim1=1, dim2=2).mul_(0.5)

        perm_adj_mat = torch.zeros_like(adj_mat)
        perm_node_mat = torch.zeros_like(node_mat)
        perm_edge_mat = torch.zeros_like(edge_mat)
        for i in range(batch_size):
            perm = permutations[i]
            # Apply the permutation to rows and columns for each matrix
            perm_adj_mat[i] = adj_mat[i][perm][:, perm]
            perm_node_mat[i] = node_mat[i][perm]
            perm_edge_mat[i] = edge_mat[i][perm][:, perm]
    
        data.adj_triu_mat = perm_adj_mat[:, self.triu_mask_adj]
        data.node_mat = perm_node_mat
        data.edge_triu_mat = perm_edge_mat[:, self.triu_mask_edge]

        return data


class DropAttributes(BaseTransform):
    """ 
    Delete given attribute from each sample to save memory
    """

    def __init__(self, attributes: List[str]) -> None:
        self.attributes = attributes

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for attr in self.attributes:
            delattr(data, attr)
        return data


def qm9_pre_filter(data: Data) -> bool:
    """ Filter samples whose graphs cannot be converted to chemically valid molecules with RDKit """
    try:
        graph_to_mol(data, includes_h=True, validate=True)
    except Exception as e:
        # print(e)
        return False
    
    # drop molecules with only one heavy atom
    hydrogen_count = data.x[:, 0].sum()
    if hydrogen_count == data.x.shape[0] - 1:
        print("Skipped Molecule")
        return False

    return True


def create_qm9_data_split(dataset) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create a training, validation and test set from the full QM9 dataset.
    """
    generator = torch.manual_seed(420)
    return random_split(dataset=dataset, lengths=[0.8, 0.1, 0.1], generator=generator)


def mol_to_image_tensor(mol) -> torch.Tensor:
    image = Draw.MolToImage(mol)
    image = np.array(image)
    # Convert to CHW format
    tensor = torch.tensor(np.transpose(image, (2, 0, 1)))
    # Add batch dimension
    return tensor.unsqueeze(0)

def smiles_to_image(smiles: str) -> torch.tensor:
    mol = Chem.MolFromSmiles(smiles)
    return mol_to_image_tensor(mol=mol)

def graph_to_mol(data: Data, includes_h: bool, validate: bool):
    # create empty molecule
    mol = Chem.RWMol()

    if includes_h:
        class_index_to_atomic_number = {
            0: 1, 1: 6, 2: 7, 3: 8, 4: 9
        }
    else:
        class_index_to_atomic_number = {
            0: 6, 1: 7, 2: 8, 3: 9
        }

    # Add atoms
    for atom_features in data.x:
        # convert the one-hot encoded atom class to the atomic number
        class_index = torch.argmax(atom_features[:5]).item()
        atomic_number = int(class_index_to_atomic_number[class_index])
        atom = Chem.Atom(atomic_number)
        mol.AddAtom(atom)

    if data.edge_index is not None:
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

    if validate:
        # Check if the molecule is chemically valid
        Chem.SanitizeMol(mol)
    
    # Convert to a standard RDKit mol object
    mol = mol.GetMol()
    return mol

def molecule_graph_data_to_image(data: Data, includes_h: bool) -> torch.Tensor:
    mol = graph_to_mol(data=data, includes_h=includes_h, validate=False)
    return mol_to_image_tensor(mol)

def create_tensorboard_writer(experiment_name: str, log_dir_root: str = "./tb_logs"):
    logdir = os.path.join(log_dir_root, experiment_name)
    os.makedirs(logdir, exist_ok=True)
    experiment_index = len(os.listdir(logdir))
    return SummaryWriter(os.path.join(logdir, str(experiment_index).zfill(3)))


def create_validation_subset_loaders(validation_dataset, subset_count, batch_size) -> List[DataLoader]:
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


def create_qm9_graph_vae_dataset(
        device: str, 
        include_hydrogen: bool, 
        refresh_data_cache: bool, 
        properties: Optional[List[str]],
        prop_norm_df: Optional[pd.DataFrame],
    ) -> QM9:

    pre_transform_list = [SelectQM9NodeFeatures(features=["atom_type"])]
    if not include_hydrogen:
        pre_transform_list.append(DropQM9Hydrogen())

    max_num_nodes = 29 if include_hydrogen else 9
    pre_transform_list += [
        AddAdjacencyMatrix(max_num_nodes=max_num_nodes),
        AddNodeAttributeMatrix(max_num_nodes=max_num_nodes),
        AddEdgeAttributeMatrix(max_num_nodes=max_num_nodes),
    ]
    pre_transform = T.Compose(pre_transform_list)

    transform_list = []
    if properties is not None:
        transform_list.append(SelectQM9TargetProperties(properties=properties))
        if prop_norm_df is not None:
            transform_list.append(NormalizeQM9Properties(properties=properties, prop_norm_df=prop_norm_df))
    transform_list.append(T.ToDevice(device=device))
    transform = T.Compose(transform_list)

    # note: when the pre_filter or pre_transform is changed, delete the data/processed folder to update the dataset
    dataset = QM9(root=GRAPH_VAE_DATA_ROOT_DIR, pre_transform=pre_transform, pre_filter=qm9_pre_filter, transform=transform)

    if refresh_data_cache:
        # remove the processed files and recreate them
        # this might be necessary when the pre_transform or the pre_filter has been updated
        shutil.rmtree(dataset.processed_dir)
        dataset = QM9(root=GRAPH_VAE_DATA_ROOT_DIR, pre_transform=pre_transform, pre_filter=qm9_pre_filter, transform=transform)

    return dataset


class AddNullEdges(BaseTransform):
    """ 
    Add null edges to the graph.
    """

    def __init__(self) -> None:
        self.edge_indices = []
        for N in range(2, 50):
            self.edge_indices.append(
                torch.tensor(list(itertools.combinations(range(N), 2))).t()
            )

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        num_nodes = data.x.shape[0]
        edge_index_full = self.edge_indices[num_nodes - 2]

        # Initialize full edge attributes with null category
        edge_attr_full = torch.zeros((edge_index_full.shape[1], 5))
        edge_attr_full[:, -1] = 1

        existing_edges = set(map(tuple, data.edge_index.t().tolist()))
        for i, edge in enumerate(edge_index_full.t()):
            edge_tuple = tuple(edge.tolist())
            if edge_tuple in existing_edges:
                index = list(existing_edges).index(edge_tuple)
                edge_attr_full[i, :-1] = data.edge_attr[index]
                edge_attr_full[i, -1] = 0

        data.edge_index_full = edge_index_full
        data.edge_attr_full = edge_attr_full

        return data


def create_qm9_mixture_vae_dataset(
        device: str,
        include_hydrogen: bool,
        refresh_data_cache: bool, 
        properties: Optional[List[str]],
        prop_norm_df: pd.DataFrame,
    ) -> QM9:

    pre_transform = SelectQM9NodeFeatures(features=["atom_type"])

    transform_list = []
    if not include_hydrogen:
        transform_list.append(DropQM9Hydrogen())
    if properties is not None:
        transform_list.extend([
            SelectQM9TargetProperties(properties=properties),
            NormalizeQM9Properties(properties=properties, prop_norm_df=prop_norm_df)
        ])
    transform_list.extend([
        AddNullEdges(),
        T.ToDevice(device=device)
    ])
    transform = T.Compose(transform_list)

    # note: when the pre_filter or pre_transform is changed, delete the data/processed folder to update the dataset
    dataset = QM9(root=MIXTURE_VAE_DATA_ROOT_DIR, pre_transform=pre_transform, pre_filter=qm9_pre_filter, transform=transform)

    if refresh_data_cache:
        # remove the processed files and recreate them
        # this might be necessary when the pre_transform or the pre_filter has been updated
        shutil.rmtree(dataset.processed_dir)
        dataset = QM9(root=MIXTURE_VAE_DATA_ROOT_DIR, pre_transform=pre_transform, pre_filter=qm9_pre_filter, transform=transform)

    return dataset


def create_or_load_property_norm_df() -> pd.DataFrame:
    # compute mean and standard deviation of property values for normalization
    norm_data_file_path = os.path.join(DATA_ROOT_DIR, "qm9_prop_norm_data.csv")
    try:
        prop_norm_df = pd.read_csv(norm_data_file_path, index_col=False)
    except FileNotFoundError:
        tmp_path = "./tmp"
        tmp_dataset = QM9(root=tmp_path)
        # compute statistics on the training set
        tmp_dataset, _, _ = create_qm9_data_split(dataset=tmp_dataset)
        y = torch.stack([data.y for data in tqdm(tmp_dataset, "Computing property statistics")])
        y_mean = y.mean(dim=0)
        y_std = y.std(dim=0)
        y_stats = torch.concat([y_mean, y_std], dim=0)
        
        prop_norm_df = pd.DataFrame(
            data=y_stats.numpy(),
            columns=QM9_PROPERTIES
        )
        prop_norm_df.to_csv(norm_data_file_path, index=False)
    
        # remove tmp directory
        shutil.rmtree(tmp_path)

    return prop_norm_df
