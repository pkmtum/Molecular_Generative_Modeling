import os
import argparse
import csv

from tqdm import tqdm
from rdkit.Chem import AllChem as Chem


def read_xyz_file(filename, csv_writer, max_heavy, property_index_list):
    file = open(filename, 'r')
    lines = file.readlines()

    atom_count = int(lines[0])

    # ignore tag and index
    properties = [lines[1].split()[2:][i] for i in property_index_list]

    # check number of heavy atoms
    num_heavy = 0
    for i in range(2, atom_count + 2):
        element_type = lines[i].split()[0]
        if element_type != "H":
            num_heavy += 1
    if num_heavy > max_heavy:
        return False

    smiles = lines[atom_count + 3].split()[1]

    try:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True, canonical=True)
    except Exception:
        print(f"Discarding smiles: {smiles}")
        return False

    csv_writer.writerow([smiles] + properties)
    return True


def main():
    parser = argparse.ArgumentParser(description="Create csv file containing smiles and molecule properties from .xyz files")
    parser.add_argument("--dir", type=str, help="Directory containing xyz files.")
    parser.add_argument("--out", default="qm9.csv", type=str, help="Output csv filename")
    parser.add_argument("--heavy", default=9, type=int, help="Maximum number of heavy atoms allowed.")
    parser.add_argument("--props", default="", type=str, help="Comma seperated properties to include in the csv file. Defaults to all.")

    args = parser.parse_args()

    properties_filter = set(args.props.split(","))

    properties = ["A", "B", "C", "mu", "alpha", "homo", "lumo",
                  "gap", "r2", "zpve", "U0", "U", "H", "G", "Cv"]

    if len(properties_filter) > 0:
        property_index_list = [i for i in range(len(properties))
                               if properties[i] in properties_filter]
        properties = [prop for prop in properties if prop in properties_filter]
    else:
        property_index_list = [i for i in range(len(properties))]

    out_csv_filename = args.out

    discarded_count = 0

    with open(out_csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        columns = ["smiles"] + properties
        writer.writerow(columns)
        total_count = 0
        directory = os.fsencode(args.dir)
        for file in tqdm(os.listdir(directory)):
            filename = os.fsdecode(file)
            if filename.endswith(".xyz"):
                filepath = os.path.join(args.dir, filename)
                if read_xyz_file(filepath, writer, args.heavy,
                                 property_index_list):
                    total_count += 1
                else:
                    discarded_count += 1

    print(f"Wrote {total_count} molecules to {out_csv_filename}.")
    print(f"Discarded {discarded_count} molecules to from the original dataset.")

if __name__ == "__main__":
    main()
