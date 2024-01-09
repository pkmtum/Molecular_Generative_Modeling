import itertools
import datetime
import os
import shutil
import argparse
from typing import Dict, Any, Union

from tqdm import tqdm
import torch
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from graph_vae.vae import GraphVAE
from data_utils import *


def create_qm9_dataset(device: str, include_hydrogen: bool, refresh_data_cache: bool) -> QM9:
    transform_list = [
        SelectQM9TargetProperties(properties=["homo", "lumo"]),
        SelectQM9NodeFeatures(features=["atom_type"]),
    ]
    if not include_hydrogen:
        transform_list.append(DropQM9Hydrogen())

    max_num_nodes = 29 if include_hydrogen else 9
    transform_list += [
        AddAdjacencyMatrix(max_num_nodes=max_num_nodes),
        AddNodeAttributeMatrix(max_num_nodes=max_num_nodes),
        AddEdgeAttributeMatrix(max_num_nodes=max_num_nodes),
        # DropAttributes(attributes=["z", "pos", "idx", "name"]),
    ]

    pre_transform = T.Compose(transform_list)
    transform = T.ToDevice(device=device)

    # note: when the pre_filter or pre_transform is changed, delete the data/processed folder to update the dataset
    dataset = QM9(root="./data", pre_transform=pre_transform, pre_filter=qm9_pre_filter, transform=transform)

    if refresh_data_cache:
        # remove the processed files and recreate them
        # this might be necessary when the pre_transform or the pre_filter has been updated
        shutil.rmtree(dataset.processed_dir)
        dataset = QM9(root="./data", pre_transform=pre_transform, pre_filter=qm9_pre_filter, transform=transform)

    return dataset


def create_dataloaders(
    train_dataset: Dataset, 
    val_dataset: Dataset, 
    test_dataset: Dataset, 
    batch_size: int,
) -> Dict[str, Union[DataLoader, List[DataLoader]]]:
    """ Create training and validation dataloaders. """
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    }

    val_subset_count = 32
    dataloaders["val_subsets"] = create_validation_subset_loaders(
        validation_dataset=val_dataset,
        subset_count=val_subset_count,
        batch_size=batch_size
    )
    return dataloaders


def train_model(
        graph_vae_model: GraphVAE,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_subset_loaders: List[DataLoader],
        tb_writer: SummaryWriter,
        epochs: int,
        start_epoch: int,
    ) -> str:
    # create checkpoint dir and unique filename
    os.makedirs("./checkpoints/", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_checkpoint = f"./checkpoints/graph_vae_{timestamp}.pt"

    # get dataloaders
    val_subset_loader_iterator = itertools.cycle(val_subset_loaders)

    validation_interval = 100

    for epoch in range(start_epoch, epochs):
        graph_vae_model.train()
        for batch_index, train_batch in enumerate(tqdm(train_loader,  desc=f"Epoch {epoch + 1} Training")):
            optimizer.zero_grad()
            
            train_elbo, train_recon_loss = graph_vae_model.elbo(x=train_batch)

            train_loss = -train_elbo
            train_loss.backward()
            optimizer.step()

            iteration = len(train_loader) * epoch + batch_index
            tb_writer.add_scalars("Loss", {"Training": train_loss.item()}, iteration)
            tb_writer.add_scalars("ELBO", {"Training": train_elbo.item()}, iteration)
            tb_writer.add_scalars("Reconstruction Loss", {"Training": train_recon_loss.item()}, iteration)
            
            if (iteration + 1) % validation_interval == 0 or iteration == 0:
                graph_vae_model.eval()
                val_loss_sum = 0
                val_elbo_sum = 0

                # Get the next subset of the validation set
                val_loader = next(val_subset_loader_iterator)
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_elbo, val_recon_loss = graph_vae_model.elbo(x=val_batch)
                        val_elbo_sum += val_elbo
                        val_loss = -val_elbo
                        val_loss_sum += val_loss
                
                val_loss = val_loss_sum / len(val_loader)
                val_elbo = val_elbo_sum / len(val_loader)
                tb_writer.add_scalars("Loss", {"Validation": val_loss.item()}, iteration)
                tb_writer.add_scalars("ELBO", {"Validation": val_elbo.item()}, iteration)
                tb_writer.add_scalars("Reconstruction Loss", {"Validation": val_recon_loss.item()}, iteration)
                
                graph_vae_model.train()

        torch.save({
                'epoch': epoch,
                'model_state_dict': graph_vae_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
            out_checkpoint
        )

    return out_checkpoint


def evaluate_model(
        graph_vae_model: GraphVAE,
        hparams: Dict[str, Union[bool, int, float]],
        device: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tb_writer: SummaryWriter,
        checkpoint_path: str,
    ) -> None:

    graph_vae_model.eval()

    log_hparams = hparams
    log_hparams.update({
        "Encoder Parameter Count": sum(p.numel() for p in graph_vae_model.encoder.parameters() if p.requires_grad),
        "Decoder Parameter Count": sum(p.numel() for p in graph_vae_model.decoder.parameters() if p.requires_grad),
    })

    # evaluate average reconstruction log-likelihood on validation set
    val_elbo_sum = 0
    val_log_likelihood_sum = 0
    for val_batch in tqdm(val_loader, desc="Evaluating Reconstruction Performance..."):
        val_elbo, val_recon_loss = graph_vae_model.elbo(x=val_batch)
        val_elbo_sum += val_elbo
        val_log_likelihood_sum -= val_recon_loss

    metrics = dict()
    metrics.update({
        "ELBO": val_elbo_sum / len(val_loader),
        "Log-likelihood": val_log_likelihood_sum / len(val_loader)
    })

    # decoding quality metrics
    train_mol_smiles = set()
    include_hydrogen = hparams["include_hydrogen"]
    for batch in tqdm(train_loader, desc="Converting training graphs to SMILES..."):
        for sample_index in range(len(batch)):
            sample = batch[sample_index]
            mol = graph_to_mol(data=sample, includes_h=include_hydrogen, validate=False)
            train_mol_smiles.add(Chem.MolToSmiles(mol))

    num_samples = 1000
    num_valid_mols = 0

    generated_mol_smiles = set()
    z, x = graph_vae_model.sample(num_samples=num_samples, device=device)
    for i in tqdm(range(num_samples), "Generating Molecules..."):
        sample_matrices = (x[0][i:i+1], x[1][i:i+1], x[2][i:i+1])
        sample_graph = graph_vae_model.output_to_graph(x=sample_matrices)
        
        try:
            mol = graph_to_mol(data=sample_graph, includes_h=include_hydrogen, validate=True)
            num_valid_mols += 1
            smiles = Chem.MolToSmiles(mol)
            if smiles in generated_mol_smiles:
                continue
            tb_writer.add_image('Generated', mol_to_image_tensor(mol=mol), global_step=len(generated_mol_smiles), dataformats="NCHW")
            generated_mol_smiles.add(Chem.MolToSmiles(mol))
        except Exception as e:
            # print(f"Invalid molecule: {e}")
            # mol = graph_to_mol(data=sample_graph, includes_h=include_hydrogen, validate=False)
            pass

    non_novel_mols = train_mol_smiles.intersection(generated_mol_smiles)
    novel_mol_count = len(generated_mol_smiles) - len(non_novel_mols)

    metrics.update({
        "Validity": num_valid_mols / num_samples,
        "Uniqueness": len(generated_mol_smiles) / num_valid_mols,
        "Novelty": novel_mol_count / len(generated_mol_smiles),  
    })
    log_hparams["checkpoint"] = checkpoint_path
    tb_writer.add_hparams(hparam_dict=log_hparams, metric_dict=metrics)


def main():
    parser = argparse.ArgumentParser("Train the GraphVAE generative model on the QM9 dataset.")
    parser.add_argument("--include_hydrogen", action="store_true", help="Include hydrogen atoms in the training data.")
    parser.add_argument("--refresh_data_cache", action="store_true", 
        help="Refresh the cached pre-processed dataset. This is required whenever the 'pre_filter' or 'pre_transform' is updated."
    )
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to resume training from.")
    parser.add_argument("--train_sample_limit", type=int, help="Maximum number of training samples to use. If unspecified, the full training set is used.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--latent_dim", type=int, default=80, help="Number of dimensions of the latent space.")
    parser.add_argument("--kl_weight", type=float, default=1e-2, help="Weight of the Kullback-Leibler divergence in the loss term.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create dataset and dataloaders
    dataset = create_qm9_dataset(
        device=device, 
        include_hydrogen=args.include_hydrogen, 
        refresh_data_cache=args.refresh_data_cache
    )
    train_dataset, val_dataset, test_dataset = create_qm9_data_split(dataset=dataset)
    if args.train_sample_limit is not None:
        train_dataset = train_dataset[:args.train_sample_limit]
    data_loaders = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
    )

    # hyperparamers
    hparams = {
        "batch_size": args.batch_size,
        "train_sample_limit": args.train_sample_limit,
        "max_num_nodes": 29 if args.include_hydrogen else 9,
        "learning_rate": 1e-3,
        "adam_beta_1": 0.5,
        "epochs": args.epochs,
        "num_node_features": dataset.num_node_features,
        "num_edge_features": dataset.num_edge_features,
        "latent_dim": args.latent_dim,  # c in the paper
        "kl_weight": args.kl_weight,
        "include_hydrogen": args.include_hydrogen,
    }

    # setup model and optimizer
    graph_vae_model = GraphVAE(hparams=hparams).to(device=device)
    optimizer = torch.optim.Adam(
        graph_vae_model.parameters(),
        lr=hparams["learning_rate"],
        betas=(hparams["adam_beta_1"], 0.999)
    )

    # load checkpoint
    if args.checkpoint is not None:
        checkpoint = checkpoint = torch.load(args.checkpoint)
        graph_vae_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    # create tensorboard summary writer
    tb_writer = create_tensorboard_writer(experiment_name="graph_vae")

    # train the model
    out_checkpoint_path = train_model(
        graph_vae_model=graph_vae_model,
        optimizer=optimizer,
        train_loader=data_loaders["train"],
        val_subset_loaders=data_loaders["val_subsets"],
        tb_writer=tb_writer,
        epochs=hparams["epochs"],
        start_epoch=start_epoch
    )

    # evaluate the model
    evaluate_model(
        graph_vae_model=graph_vae_model,
        hparams=hparams,
        device=device,
        train_loader=data_loaders["train"],
        val_loader=data_loaders["val"],
        tb_writer=tb_writer,
        checkpoint_path=out_checkpoint_path,
    )


if __name__ == "__main__":
    main()