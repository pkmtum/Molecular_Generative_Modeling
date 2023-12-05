import argparse
import json

from tqdm import tqdm
import torch
from torch_geometric.datasets import QM9

def main():
    parser = argparse.ArgumentParser(
        description="Create a mapping from the one-hot encoded atomic number in the torch-geometric QM9 to the actual atomic number."
    )
    parser.add_argument("--out", default="qm9_hot_atomic_number.json", type=str, help="Output json file.")

    args = parser.parse_args()
    out_json_filename = args.out
    
    dataset = QM9(root="./data")

    one_hot_atom_num_map = {}
    max_atom_count = 0

    for data in tqdm(dataset):
        max_atom_count = max(max_atom_count, data.x.shape[0])
        for atom_features in data.x:
            one_hot = atom_features[:5]
            atomic_number = atom_features[5].item()
            class_index = torch.argmax(one_hot).item()
            one_hot_atom_num_map[class_index] = int(atomic_number)
        
    print(f"Found {len(one_hot_atom_num_map)} unique atoms!")
    print(f"The largest molecule has {max_atom_count} atoms!")
    json_str = json.dumps(one_hot_atom_num_map, indent=4)
    with open(out_json_filename, "w") as outfile:
        outfile.write(json_str)

if __name__ == "__main__":
    main()