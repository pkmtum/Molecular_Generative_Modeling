from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch.utils.data import random_split
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import argparse


class GCN(torch.nn.Module):
    def __init__(self, num_node_features: int, num_targets: int):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.fc1 = nn.Linear(32, num_targets)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        return x


def run_training(model, optimizer, train_dataset, val_dataset, batch_size, device):
    writer = SummaryWriter()
    epochs = 10

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

    print("Training...")
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            optimizer.zero_grad()
            sample = batch.to(device)
            model_output = model(sample)
            loss = F.mse_loss(model_output, sample.y)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_train_loss /= len(train_loader.dataset)
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)

        model.eval()
        epoch_val_loss = 0
        for batch in tqdm(val_loader, desc=f"Epoch {epoch} Validator"):
            sample = batch.to(device)
            model_output = model(sample)
            loss = F.mse_loss(model_output, sample.y)
            epoch_val_loss += loss.item()

        writer.add_scalars()

def main():
    parser = argparse.ArgumentParser(description="Training a graph property predictor on the QM9 dataset.")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    transform = T.Compose([
        T.NormalizeFeatures(attrs=["y"]),
    ])
    dataset = QM9(root="./data", transform=transform)
    train_dataset, val_dataset = random_split(dataset=dataset, lengths=[0.9, 0.1], generator=torch.manual_seed(420))

    model = GCN(num_node_features=dataset.num_node_features, num_targets=dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    batch_size = args.batch_size

    run_training(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        device=device
    )

if __name__ == "__main__":
    main()