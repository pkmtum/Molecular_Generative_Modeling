import argparse
import functools
import json
from typing import Dict, Any

import torch
import torch.nn.functional as F
import torch.distributions as D
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
import torch_geometric.utils as pyg_utils

from tqdm import tqdm
import networkx as nx
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from data_utils import *
from mixture_model.decoder import MixtureModelDecoder
from mixture_model.encoder import MixtureModelEncoder


def kl_divergence_gaussian(mu_q, sigma_q, mu_p, sigma_p):
    """
    KL(q||p) where q and p are Gaussian distribution is diagonal covariance matrices.
    """
    epsilon = 1e-6
    sigma_q = torch.clamp(sigma_q, min=epsilon)
    sigma_p = torch.clamp(sigma_p, min=epsilon)
    return (torch.log(sigma_p / sigma_q) + ((sigma_q ** 2 + (mu_q - mu_p) ** 2)) / (2 * (sigma_p ** 2)) - 0.5).sum(dim=1)


def kl_divergence_categorical(pi_q, pi_p):
    """
    KL(q||p) where q and p and categorical distributions.
    """
    epsilon = 1e-6
    pi_q = torch.clamp(pi_q, min=epsilon)
    pi_p = torch.clamp(pi_p, min=epsilon)
    return (pi_q * torch.log(pi_q / pi_p)).sum(dim=1)


def monotonic_cosine_schedule(iteration: int, start_iteration: int, end_iteration: int) -> float:
    length = end_iteration - start_iteration
    x = min(max(iteration - start_iteration, 0) / length, 1)
    return 0.5 * (1 + math.cos((1 + x) * math.pi)) * 0.9999 + 0.0001


def train_model(
        encoder_model: MixtureModelEncoder,
        decoder_model: MixtureModelDecoder,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        tb_writer: SummaryWriter,
        hparams: Dict[str, Any],
    ):
    epochs = hparams["epochs"]
    eta_kl_weight = hparams["eta_kl_weight"]
    c_kl_weight = hparams["c_kl_weight"]
    z_kl_weight = hparams["z_kl_weight"]

    encoder_model.train()
    decoder_model.train()

    # parameters for gumbel softmax temperature annealing from https://arxiv.org/abs/1611.01144
    N = 500
    r = 1e-4
    tau = 1.0
    decoder_model.set_gumbel_softmax_temperature(temperature=tau)

    total_iteration_count = len(train_loader) * epochs
    start_iteration = int(total_iteration_count * 0.25)
    end_iteration = int(total_iteration_count * 0.75)
    kl_schedule_func = functools.partial(monotonic_cosine_schedule, start_iteration=start_iteration, end_iteration=end_iteration)

    for epoch in range(epochs):

        for batch_index, train_batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} Training")):

            iteration = len(train_loader) * epoch + batch_index

            kl_weight = kl_schedule_func(iteration)

            # anneal the gumbel softmax temperature
            if (iteration + 1) % N == 0:
                tau = max(0.5, math.exp(-r * iteration))
                decoder_model.set_gumbel_softmax_temperature(tau)

            optimizer.zero_grad()

            train_z_mu, train_z_sigma, train_eta_mu, train_eta_sigma = encoder_model(train_batch)
            train_z = torch.randn_like(train_z_mu) * train_z_sigma + train_z_mu
            train_num_atoms = torch.bincount(train_batch.batch)
            train_reconstruction = decoder_model.decode_z(train_z, train_num_atoms)
            train_target_x = torch.argmax(train_batch.x, dim=1)

            # atom reconstruction loss
            train_loss = F.cross_entropy(input=train_reconstruction.x, target=train_target_x, reduction="sum")
            
            # bond reconstruction loss
            train_target_edge_attr = torch.argmax(train_batch.edge_attr_full, dim=1)
            train_loss += F.cross_entropy(
                input=train_reconstruction.edge_attr,
                target=train_target_edge_attr,
                reduction="sum",
            )
            train_loss /= len(train_batch)
            train_log_likelihood = -train_loss

            # eta KL-Divergence
            eta_kl_divergence = kl_divergence_gaussian(
                mu_q=train_eta_mu,
                sigma_q=train_eta_sigma,
                mu_p=decoder_model.eta_mu,
                sigma_p=torch.exp(torch.clamp(decoder_model.eta_log_sigma, -30, 20))
            ).mean()  # mean over batch
            train_loss += eta_kl_divergence * eta_kl_weight * kl_weight

            # cluster KL-Divergence
            train_eta = torch.randn_like(train_eta_mu) * train_eta_sigma + train_eta_mu
            train_pi_p = decoder_model.decode_eta(train_eta)
            train_pi_p = torch.repeat_interleave(train_pi_p, train_num_atoms, dim=0)
            train_z_log_likelihoods = D.Normal(
                loc=decoder_model.cluster_means, 
                scale=torch.exp(torch.clamp(decoder_model.cluster_log_sigmas, -30, 20))
            ).log_prob(train_z.unsqueeze(1)).sum(dim=2)
            train_weighted_z_log_likelihood = train_z_log_likelihoods + torch.log(train_pi_p)
            train_z_log_responsibilities = train_weighted_z_log_likelihood - torch.logsumexp(train_weighted_z_log_likelihood, dim=1, keepdim=True)
            train_pi_q = torch.exp(train_z_log_responsibilities)  # posterior distribution of the cluster indicators
            cluster_kl_divergence = kl_divergence_categorical(
                pi_q=train_pi_q,
                pi_p=train_pi_p,
            )
            # sum over all atoms in each molecule
            cluster_kl_divergence = scatter(cluster_kl_divergence, train_batch.batch, dim=0, reduce='sum')
            # mean over batches
            cluster_kl_divergence = cluster_kl_divergence.mean()
            train_loss += cluster_kl_divergence * c_kl_weight

            # z KL-Divergence
            sigma_pc = torch.exp(torch.clamp(decoder_model.cluster_log_sigmas, -30, 20))
            z_kl_divergence = kl_divergence_gaussian(
                mu_q=train_z_mu.unsqueeze(-1),
                sigma_q=train_z_sigma.unsqueeze(-1),
                mu_p=decoder_model.cluster_means.transpose(1, 2),
                sigma_p=sigma_pc.transpose(1, 2)
            )
            z_kl_divergence = (z_kl_divergence * train_pi_q).sum(dim=1)
            # sum over all nodes in each molecule and average over batch
            z_kl_divergence = scatter(z_kl_divergence, train_batch.batch, dim=0, reduce='sum').mean()
            train_loss += z_kl_divergence * z_kl_weight * kl_weight

            train_loss.backward()
            optimizer.step()

            # log to tensorboard
            tb_writer.add_scalars("Loss", {"Training": train_loss.item()}, iteration)
            tb_writer.add_scalars("Log-Likelihood", {"Training": train_log_likelihood.item()}, iteration)
            tb_writer.add_scalar("Eta KL-Divergence", eta_kl_divergence.item(), iteration)
            tb_writer.add_scalar("Cluster KL-Divergence", cluster_kl_divergence.item(), iteration)
            tb_writer.add_scalar("Z KL-Divergence", z_kl_divergence.item(), iteration)


def get_batch_item(batch_data: Data, i: int):
    node_mask = batch_data.batch == i

    offsets = torch.cumsum(torch.bincount(batch_data.batch), dim=0)

    edge_mask = (batch_data.edge_index[0] < offsets[i]) & (batch_data.edge_index[1] < offsets[i])
    if i > 0:
        edge_mask &= (batch_data.edge_index[0] >= offsets[i - 1]) & (batch_data.edge_index[1] >= offsets[i - 1])

    # remove non-existent edges
    edge_mask &= batch_data.edge_attr.argmax(dim=1) < 4
    edge_attr = batch_data.edge_attr[edge_mask][:,:-1]

    # adjust edge index based on batch index
    edge_index = batch_data.edge_index[:, edge_mask]
    if i > 0:
        edge_index -= offsets[i - 1]

    return Data(
        x=batch_data.x[node_mask],
        edge_index=edge_index,
        edge_attr=edge_attr
    )


def main():
    parser = argparse.ArgumentParser("Train the structured generative model on the QM9 dataset.")
    parser.add_argument("--include_hydrogen", action="store_true", help="Include hydrogen atoms in the training data.")
    parser.add_argument("--use_cached_dataset", action="store_true", 
        help="Use the cached pre-processed dataset for training to avoid time consuming pre-processing before each training run."
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--num_clusters", type=int, default=32, help="Number of clusters in the gaussian mixture.")
    parser.add_argument("--eta_dim", type=int, help="Dimension of the latent variable eta.")
    parser.add_argument("--z_dim", type=int, default=8)
    parser.add_argument("--cluster_mlp_hidden_dim", type=int, default=256,
        help="Number of dimensions of the hidden layer in the cluster MLP. If zero, the mapping from eta to pi is just a softmax."
    )
    parser.add_argument("--bond_type_mlp_hidden_dim", type=int, default=256,
        help="Number of dimensions of the hidden layer in the bond type MLP. If zero, the mapping from a pair of latent vectors to a bond type is a linear operation."
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--eta_kl_weight", type=float, default=1.0, help="Weight of the KL-divergence over eta in the loss.")
    parser.add_argument("--c_kl_weight", type=float, default=1.0, help="Weight of the KL-divergence over the cluster probabilities in the loss.")
    parser.add_argument("--z_kl_weight", type=float, default=0.1, help="Weight of the KL-divergence over the z in the loss.")
    parser.add_argument("--logdir", type=str, default="mixture_model", help="Name of the Tensorboard logging directory.")
    args = parser.parse_args()

    if args.eta_dim is None:
        eta_dim = args.num_clusters
    else:
        eta_dim = args.eta_dim

    if args.cluster_mlp_hidden_dim == 0 and eta_dim != args.num_clusters:
        raise ValueError(f'eta_dim ({args.eta_dim}) must equal num_cluster ({args.num_clusters}) when cluster_mlp_hidden_dim = 0')


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create data set and data loaders
    prop_norm_df = create_or_load_property_norm_df()
    dataset = create_qm9_mixture_vae_dataset(
        device=device, 
        include_hydrogen=args.include_hydrogen,
        refresh_data_cache=False,
        properties=None,
        prop_norm_df=prop_norm_df
    )
    train_dataset, val_dataset, _ = create_qm9_data_split(dataset=dataset)

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    hparams = {
        "include_hydrogen": args.include_hydrogen,
        "eta_dim": eta_dim,
        "num_clusters": args.num_clusters,
        "z_dim": args.z_dim,
        "num_atom_types": dataset.num_node_features,
        "num_bond_types": dataset.num_edge_features,
        "eta_kl_weight": args.eta_kl_weight,
        "c_kl_weight": args.c_kl_weight,
        "z_kl_weight": args.z_kl_weight,
        "cluster_mlp_hidden_dim": args.cluster_mlp_hidden_dim,
        "bond_type_mlp_hidden_dim": args.bond_type_mlp_hidden_dim,
        "batch_size": batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
    }

    encoder_model = MixtureModelEncoder(hparams=hparams).to(device)
    decoder_model = MixtureModelDecoder(hparams=hparams).to(device)

    optimizer = torch.optim.Adam(
        params=list(encoder_model.parameters()) + list(decoder_model.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-6
    )

    tb_writer = create_tensorboard_writer(experiment_name=args.logdir)

    train_model(
        encoder_model=encoder_model,
        decoder_model=decoder_model,
        optimizer=optimizer,
        train_loader=train_loader,
        tb_writer=tb_writer,
        hparams=hparams,
    )

    # create set of training data SMILES to check the novelty
    train_mol_smiles = set()
    smiles_file_path = os.path.join(MIXTURE_VAE_DATA_ROOT_DIR, "qm9_train_smiles.json")
    try:
        with open(smiles_file_path, "r") as file:
            train_mol_smiles = set(json.load(file))
    except FileNotFoundError:
        for batch in tqdm(train_loader, desc="Converting training graphs to SMILES"):
            for sample_index in range(len(batch)):
                sample = batch[sample_index]
                mol = graph_to_mol(data=sample, includes_h=args.include_hydrogen, validate=False)
                train_mol_smiles.add(Chem.MolToSmiles(mol))

        # write SMILES strings to the json file so can just load them the next time
        with open(smiles_file_path, "w") as file:
            json.dump(list(train_mol_smiles), file)

    # evaluate the model by generating 1000 molecules
    decoder_model.eval()
    num_generated_mols = 1000
    num_atoms = torch.tensor([9] * num_generated_mols, dtype=torch.int64, device=device)
    with torch.no_grad():
        data = decoder_model.sample(num_atoms, device)

    # TODO: evaluate uniqueness & novelty
    num_valid_mols = 0
    num_connected_graphs = 0
    generated_mol_smiles = set()
    for i in tqdm(range(num_generated_mols)):
        graph = get_batch_item(data, i)
        mol = graph_to_mol(data=graph, includes_h=args.include_hydrogen, validate=False)
        tb_writer.add_image('Generated', mol_to_image_tensor(mol=mol), global_step=i, dataformats="NCHW")

        if nx.is_connected(pyg_utils.to_networkx(graph, to_undirected=True)):
            num_connected_graphs += 1
        else:
            # graph is not connected; try to decode again
            continue

        try:
            mol = graph_to_mol(data=graph, includes_h=args.include_hydrogen, validate=True)
        except:
            continue
        tb_writer.add_image('Generated Valid', mol_to_image_tensor(mol=mol), global_step=i, dataformats="NCHW")
        num_valid_mols += 1

        smiles = Chem.MolToSmiles(mol)
        if smiles not in generated_mol_smiles:
            generated_mol_smiles.add(Chem.MolToSmiles(mol))

    non_novel_mols = train_mol_smiles.intersection(generated_mol_smiles)
    novel_mol_count = len(generated_mol_smiles) - len(non_novel_mols)

    tb_writer.add_hparams(
        hparam_dict=hparams,
        metric_dict={
            "Connectedness": num_connected_graphs / num_generated_mols,
            "Validity": num_valid_mols / num_generated_mols,
            "Uniqueness": len(generated_mol_smiles) / max(num_valid_mols, 1),
            "Novelty": novel_mol_count / max(len(generated_mol_smiles), 1),
        }
    )
    
    # TODO: checkpoint saving and loading -> MixtureModel autoencoder
    # TODO: parameter counts


if __name__ == "__main__":
    main()