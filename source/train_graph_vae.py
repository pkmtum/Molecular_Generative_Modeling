import itertools
import datetime
import os
import argparse
import json
from typing import Dict, Any, Union
import math
import functools

import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch_geometric.utils as pyg_utils

import networkx as nx

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from graph_vae.vae import GraphVAE
from data_utils import *


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
    return dataloaders


def sigmoid_schedule(epoch: int, start: int, slope: float) -> float:
    return 1 / (1 + np.exp(slope * (start - epoch)))


def cyclic_linear_schedule(iteration: int, cycle_length: int) -> float:
    linear_length = cycle_length // 2
    return min(1, (iteration % cycle_length) / linear_length) 


def cyclic_cosine_schedule(iteration: int, cycle_length: int) -> float:
    cosine_length = cycle_length // 2
    return 0.5 * (1 + math.cos((1 + min(1, (iteration % cycle_length) / cosine_length)) * math.pi))


def monotonic_cosine_schedule(iteration: int, start_iteration: int, end_iteration: int) -> float:
    length = end_iteration - start_iteration
    x = min(max(iteration - start_iteration, 0) / length, 1)
    return 0.5 * (1 + math.cos((1 + x) * math.pi))


def train_model(
        graph_vae_model: GraphVAE,
        optimizer: torch.optim.Optimizer,
        hparams: Dict[str, Union[bool, int, float]],
        train_loader: DataLoader,
        val_loader: DataLoader,
        tb_writer: SummaryWriter,
        epochs: int,
        start_epoch: int,
    ) -> str:
    # create checkpoint dir and unique filename
    os.makedirs("./checkpoints/", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_checkpoint = f"./checkpoints/graph_vae_{timestamp}.pt"

    properties = hparams["properties"]
    predict_properties = len(properties) > 0

    kl_schedule_type = hparams["kl_schedule"]
    kl_schedule_func = None
    if kl_schedule_type == "constant":
        kl_schedule_func = lambda x: 1
    elif kl_schedule_type == "cyclical":
        total_iteration_count = len(train_loader) * epochs
        num_cycles = 4
        cycle_length = math.ceil(total_iteration_count / num_cycles)
        kl_schedule_func = functools.partial(cyclic_cosine_schedule, cycle_length=cycle_length)
    elif kl_schedule_type == "monotonic":
        total_iteration_count = len(train_loader) * epochs
        start_iteration = int(total_iteration_count * 0.25)
        end_iteration = int(total_iteration_count * 0.75)
        kl_schedule_func = functools.partial(monotonic_cosine_schedule, start_iteration=start_iteration, end_iteration=end_iteration)

    kl_weight_scale = hparams["kl_weight"]

    nll_loss_func = nn.GaussianNLLLoss(full=True)
    best_val_loss = 100000

    for epoch in range(start_epoch, start_epoch + epochs):
        graph_vae_model.train()

        train_loss_sum = 0
        train_elbo_sum = 0
        train_recon_loss_sum = 0
        train_kl_divergence_sum = 0
        train_property_loss_sum = 0
        train_mean_property_std_sum = 0

        for batch_index, train_batch in enumerate(tqdm(train_loader,  desc=f"Epoch {epoch + 1} Training")):
            optimizer.zero_grad()

            iteration = len(train_loader) * epoch + batch_index

            kl_weight = kl_schedule_func(iteration) * kl_weight_scale
            
            train_model_ouput = graph_vae_model(train_batch)
            train_recon, mu, sigma = train_model_ouput[:3]

            train_target = (train_batch.adj_triu_mat, train_batch.node_mat, train_batch.edge_triu_mat)

            train_recon_loss = graph_vae_model.reconstruction_loss(input=train_recon, target=train_target)
            train_kl_divergence = GraphVAE.kl_divergence(mu=mu, sigma=sigma)
            train_loss = train_recon_loss + kl_weight * train_kl_divergence
            train_elbo = -train_loss

            if predict_properties:
                train_pred_y_mu = train_model_ouput[3]
                train_pred_y_sigma = train_model_ouput[4]
                train_pred_y_var = train_pred_y_sigma * train_pred_y_sigma
                train_property_loss = nll_loss_func(train_pred_y_mu, train_batch.y, train_pred_y_var)
                train_loss += train_property_loss

                train_mean_property_std = train_pred_y_sigma.mean()

            train_loss.backward()
            optimizer.step()    

            train_loss_sum += train_loss.item()
            train_elbo_sum += train_elbo.item()
            train_recon_loss_sum += train_recon_loss.item()
            train_kl_divergence_sum += train_kl_divergence.item()
            if predict_properties:
                train_property_loss_sum += train_property_loss.item()
                train_mean_property_std_sum += train_mean_property_std.item()

            tb_writer.add_scalar("KL Weight", kl_weight, iteration)
                

        # tensorboard logging
        tb_writer.add_scalars("Loss", {"Training": train_loss_sum / len(train_loader)}, epoch)
        tb_writer.add_scalars("ELBO", {"Training": train_elbo_sum / len(train_loader)}, epoch)
        tb_writer.add_scalars("Reconstruction Loss", {"Training": train_recon_loss_sum / len(train_loader)}, epoch)
        tb_writer.add_scalars("KL Divergence", {"Training": train_kl_divergence_sum / len(train_loader)}, epoch)
        if predict_properties:
            tb_writer.add_scalars("Property Regression Loss", {"Training": train_property_loss_sum / len(train_loader)}, epoch)
            tb_writer.add_scalars("Mean Property Std", {"Training": train_mean_property_std_sum / len(train_loader)}, epoch)

        # validation
        graph_vae_model.eval()
        val_loss_sum = 0
        val_elbo_sum = 0
        val_recon_loss_sum = 0
        val_kl_divergence_sum = 0
        val_property_loss_sum = 0
        val_mean_property_std_sum = 0
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="Validation"):
                model_output = graph_vae_model(val_batch)
                val_recon, mu, sigma = model_output[:3]

                val_target = (val_batch.adj_triu_mat, val_batch.node_mat, val_batch.edge_triu_mat)

                val_recon_loss = graph_vae_model.reconstruction_loss(input=val_recon, target=val_target)
                val_kl_divergence = GraphVAE.kl_divergence(mu=mu, sigma=sigma)
                val_loss = val_recon_loss + kl_weight * val_kl_divergence

                val_elbo_sum -= val_loss
                val_recon_loss_sum += val_recon_loss
                val_kl_divergence_sum += val_kl_divergence

                if predict_properties:
                    val_pred_y_mu = model_output[3]
                    val_pred_y_sigma = model_output[4]
                    val_pred_y_var = val_pred_y_sigma * val_pred_y_sigma
                    val_property_loss = nll_loss_func(val_pred_y_mu, val_batch.y, val_pred_y_var)
                    val_property_loss_sum += val_property_loss
                    val_loss += val_property_loss
                    val_mean_property_std_sum += val_pred_y_sigma.mean()
                    
                val_loss_sum += val_loss
                    
            val_loss = val_loss_sum.item() / len(val_loader)
            val_elbo = val_elbo_sum.item() / len(val_loader)
            val_recon_loss = val_recon_loss_sum.item() / len(val_loader)
            val_kl_divergence = val_kl_divergence_sum.item() / len(val_loader)

            tb_writer.add_scalars("Loss", {"Validation": val_loss}, epoch)
            tb_writer.add_scalars("ELBO", {"Validation": val_elbo}, epoch)
            tb_writer.add_scalars("Reconstruction Loss", {"Validation": val_recon_loss}, epoch)
            tb_writer.add_scalars("KL Divergence", {"Validation": val_kl_divergence}, epoch)
            if predict_properties:
                val_property_loss = val_property_loss_sum.item() / len(val_loader)
                tb_writer.add_scalars("Property Regression Loss", {"Validation": val_property_loss}, epoch)
                val_property_std = val_mean_property_std_sum.item() / len(val_loader)
                tb_writer.add_scalars("Mean Property Std", {"Validation": val_property_std}, epoch)


        graph_vae_model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                    "epoch": epoch,
                    "model_state_dict": graph_vae_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "hparams": hparams,
                },
                out_checkpoint
            )
            print(f"Saved GraphVAE training checkpoint to {out_checkpoint}")

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

    properties = hparams["properties"]
    include_hydrogen = hparams["include_hydrogen"]

    log_hparams["properties"] = ",".join(hparams["properties"])

    kl_weight = hparams["kl_weight"]

    nll_loss_func = nn.GaussianNLLLoss(full=True)

    # evaluate average reconstruction log-likelihood on validation set
    val_elbo_sum = 0
    val_log_likelihood_sum = 0
    property_prediction_loss = 0
    property_mae_list = []
    property_std_list = []
    with torch.no_grad():
        for val_index, val_batch in enumerate(tqdm(val_loader, desc="Evaluating Reconstruction Performance")):
            model_output = graph_vae_model(val_batch)
            val_recon, mu, sigma = model_output[:3]

            if len(properties) > 0:
                val_pred_y_mu = model_output[3]
                val_pred_y_sigma = model_output[4]
                val_pred_y_var = val_pred_y_sigma * val_pred_y_sigma
                property_prediction_loss += nll_loss_func(val_pred_y_mu, val_batch.y, val_pred_y_var)

                denorm_target_properties = graph_vae_model.denormalize_properties(val_batch.y)
                denorm_pred_properties = graph_vae_model.denormalize_properties(val_pred_y_mu)
                property_mae = (denorm_pred_properties - denorm_target_properties).abs()
                property_mae_list.append(property_mae)

                denorm_property_std = graph_vae_model.denormalize_properties_std(val_pred_y_sigma)
                property_std_list.append(denorm_property_std)
            
            val_target = (val_batch.adj_triu_mat, val_batch.node_mat, val_batch.edge_triu_mat)
            
            # plot input and reconstruction graphs in first batch to tensorboard
            if val_index == 0:
                batch_size = val_batch.adj_triu_mat.shape[0]
                for i in range(batch_size):
                    input_mol = graph_to_mol(data=val_batch[i], includes_h=include_hydrogen, validate=False)
                    x = (val_recon[0][i:i+1], val_recon[1][i:i+1], val_recon[2][i:i+1])
                    recon_graph = graph_vae_model.output_to_graph(x=x, stochastic=False)
                    recon_mol = graph_to_mol(data=recon_graph, includes_h=include_hydrogen, validate=False)

                    tb_writer.add_image('Validation Input', mol_to_image_tensor(mol=input_mol), global_step=i, dataformats="NCHW")
                    tb_writer.add_image('Validation Reconstruction', mol_to_image_tensor(mol=recon_mol), global_step=i, dataformats="NCHW")

            val_recon_loss = graph_vae_model.reconstruction_loss(input=val_recon, target=val_target)
            val_loss = val_recon_loss + GraphVAE.kl_divergence(mu=mu, sigma=sigma) * kl_weight

            val_elbo_sum -= val_loss
            val_log_likelihood_sum -= val_recon_loss

    metrics = dict()
    metrics.update({
        "ELBO": val_elbo_sum / len(val_loader),
        "Log-likelihood": val_log_likelihood_sum / len(val_loader),
        "Property Prediction Gaussian NLL": property_prediction_loss / len(val_loader),
    })
    # this is technically a metric but only hparams can be strings
    if len(properties) > 0:
        log_hparams["Property Unnormalized MAE"] = (
            ", ".join([f"{x:.4f}" for x in torch.cat(property_mae_list, dim=0).mean(0).tolist()])
        )
    else:
        log_hparams["Property Unnormalized MAE"] = ""

    if len(properties) > 0:
        log_hparams["Property Unnormalized Std"] = (
            ", ".join([f"{x:.4f}" for x in torch.cat(property_std_list, dim=0).mean(0).tolist()])
        )
    else:
        log_hparams["Property Unnormalized Std"] = ""

    # decoding quality metrics
    train_mol_smiles = set()
    include_hydrogen = hparams["include_hydrogen"]

    smiles_file_path = os.path.join(DATA_ROOT_DIR, "qm9_train_smiles.json")
    try:
        with open(smiles_file_path, "r") as file:
            train_mol_smiles = set(json.load(file))
    except FileNotFoundError:
        for batch in tqdm(train_loader, desc="Converting training graphs to SMILES"):
            for sample_index in range(len(batch)):
                sample = batch[sample_index]
                mol = graph_to_mol(data=sample, includes_h=include_hydrogen, validate=False)
                train_mol_smiles.add(Chem.MolToSmiles(mol))

        # write SMILES strings to the json file so can just load them the next time
        with open(smiles_file_path, "w") as file:
            json.dump(list(train_mol_smiles), file)

    num_samples = 1000
    num_valid_mols = 0
    num_connected_graphs = 0

    use_stochastic_decoding = hparams["stochastic_decoding"]
    if use_stochastic_decoding:
        max_decode_attempts = hparams["max_decode_attempts"]
    else:
        max_decode_attempts = 1

    total_decode_attempts = 0
    generated_mol_smiles = set()
    z, x = graph_vae_model.sample(num_samples=num_samples, device=device)
    for i in tqdm(range(num_samples), "Generating Molecules"):
        sample_matrices = (x[0][i:i+1], x[1][i:i+1], x[2][i:i+1])

        # attempt to decode multiply time until we have both a connected graph and a valid molecule
        for _ in range(max_decode_attempts):
            sample_graph = graph_vae_model.output_to_graph(x=sample_matrices, stochastic=use_stochastic_decoding)
            total_decode_attempts += 1

            # check if the generated graph is connected
            if nx.is_connected(pyg_utils.to_networkx(sample_graph, to_undirected=True)):
                num_connected_graphs += 1
            else:
                # graph is not connected; try to decode again
                continue
        
            try:
                mol = graph_to_mol(data=sample_graph, includes_h=include_hydrogen, validate=True)
            except Exception as e:
                # Molecule is invalid; try to decode again
                continue

            # Molecule is valid
            num_valid_mols += 1
            smiles = Chem.MolToSmiles(mol)
            if smiles not in generated_mol_smiles:
                tb_writer.add_image('Generated', mol_to_image_tensor(mol=mol), global_step=len(generated_mol_smiles), dataformats="NCHW")
                generated_mol_smiles.add(Chem.MolToSmiles(mol))
            break


    non_novel_mols = train_mol_smiles.intersection(generated_mol_smiles)
    novel_mol_count = len(generated_mol_smiles) - len(non_novel_mols)

    metrics.update({
        "Mean Decode Attempts": total_decode_attempts / num_samples,
        "Connectedness": num_connected_graphs / total_decode_attempts,
        "Validity": num_valid_mols / total_decode_attempts,
        "Uniqueness": len(generated_mol_smiles) / num_valid_mols,
        "Novelty": novel_mol_count / len(generated_mol_smiles),  
    })
    log_hparams["checkpoint"] = checkpoint_path
    tb_writer.add_hparams(hparam_dict=log_hparams, metric_dict=metrics)


def main():
    parser = argparse.ArgumentParser("Train the GraphVAE generative model on the QM9 dataset.")
    parser.add_argument("--include_hydrogen", action="store_true", help="Include hydrogen atoms in the training data.")
    parser.add_argument("--use_cached_dataset", action="store_true", 
        help="Use the cached pre-processed dataset for training to avoid time consuming pre-processing before each training run."
    )
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to resume training from.")
    parser.add_argument("--train_sample_limit", type=int, help="Maximum number of training samples to use. If unspecified, the full training set is used.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--latent_dim", type=int, default=128, help="Number of dimensions of the latent space.")
    parser.add_argument("--kl_schedule", type=str, choices=["constant", "cyclical", "monotonic"], default="monotonic", help="Type of annealing schedule to use for the weight of the KL divergence.")
    parser.add_argument("--stochastic_decoding", action="store_true", help="Decode molecules stochastically.")
    parser.add_argument("--max_decode_attempts", type=int, default=10, help="Maximum number of stochastic decoding attempt until a valid molecule is decoded.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--properties", type=str, help="Properties to predict from the latent space.")
    parser.add_argument("--kl_weight", type=float, default=1e-2, help="Weight of the KL-Divergence loss term.")
    parser.add_argument("--logdir", type=str, default="graph_vae_dev_x", help="Name of the Tensorboard logging directory.")
    parser.add_argument("--property_latent_dim", type=int, help="Size of the portion of the latent space used for property prediction.")
    parser.add_argument("--prop_net_hidden_dim", type=int, default=100, help="Number of neurons in the hidden layers of the property predictor.")
    args = parser.parse_args()

    if args.property_latent_dim is None:
        property_latent_dim = args.latent_dim
    else:
        property_latent_dim = args.property_latent_dim

    # --properties=homo,lumo,r2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.properties is not None:
        properties = args.properties.split(",")
        for property in properties:
            if property not in QM9_PROPERTIES:
                raise ValueError(f"Property {property} does not exist in the QM9 dataset.")
    elif args.properties == "all":
        properties = None
    else:
        properties = []


    prop_norm_df = create_or_load_property_norm_df()
    
    # create dataset and dataloaders
    dataset = create_qm9_dataset(
        device=device, 
        include_hydrogen=args.include_hydrogen, 
        refresh_data_cache=not args.use_cached_dataset,
        properties=properties,
        prop_norm_df=prop_norm_df
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
        "learning_rate": args.learning_rate,
        "adam_beta_1": 0.5,
        "epochs": args.epochs,
        "num_node_features": dataset.num_node_features,
        "num_edge_features": dataset.num_edge_features,
        "latent_dim": args.latent_dim,  # c in the paper
        "kl_schedule": args.kl_schedule,
        "include_hydrogen": args.include_hydrogen,
        "stochastic_decoding": args.stochastic_decoding,
        "max_decode_attempts": args.max_decode_attempts,
        "properties": properties,
        "kl_weight": args.kl_weight,
        "property_latent_dim": property_latent_dim,
        "prop_net_hidden_dim": args.prop_net_hidden_dim
    }

    # setup model and optimizer
    graph_vae_model = GraphVAE(hparams=hparams, prop_norm_df=prop_norm_df).to(device=device)
    optimizer = torch.optim.Adam(
        graph_vae_model.parameters(),
        lr=hparams["learning_rate"],
        betas=(hparams["adam_beta_1"], 0.999)
    )

    # load checkpoint
    if args.checkpoint is not None:
        graph_vae_model = GraphVAE.from_pretrained(args.checkpoint).to(device)
        checkpoint = torch.load(args.checkpoint)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        hparams = checkpoint["hparams"]
        start_epoch = checkpoint['epoch'] + 1
        hparams["epochs"] = args.epochs + start_epoch
        hparams["stochastic_decoding"] = args.stochastic_decoding
        hparams["max_decode_attempts"] = args.max_decode_attempts
    else:
        start_epoch = 0

    # create tensorboard summary writer
    tb_writer = create_tensorboard_writer(experiment_name=args.logdir)

    if args.epochs > 0:
        # train the model
        out_checkpoint_path = train_model(
            graph_vae_model=graph_vae_model,
            optimizer=optimizer,
            hparams=hparams,
            train_loader=data_loaders["train"],
            val_loader=data_loaders["val"],
            tb_writer=tb_writer,
            epochs=args.epochs,
            start_epoch=start_epoch
        )
    elif args.checkpoint is not None:
        out_checkpoint_path = args.checkpoint
    else:
        # when epochs = 0 we are only evaluating the model
        # thus we need a checkpoint to load from
        raise ValueError("Missing trainng checkpoint!")

    # load best model
    graph_vae_model = GraphVAE.from_pretrained(out_checkpoint_path).to(device)
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