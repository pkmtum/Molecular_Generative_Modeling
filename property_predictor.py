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
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_targets)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def run_training(model, optimizer, train_dataset, val_dataset, batch_size, device, epochs):
    writer = SummaryWriter()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Training...")
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        count = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            optimizer.zero_grad()
            sample = batch.to(device)

            print(sample.y)
            print(sample.y.shape)

            model_output = model(sample)
            loss = F.mse_loss(model_output, sample.y)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)

        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                sample = batch.to(device)
                model_output = model(sample)
                loss = F.mse_loss(model_output, sample.y)

                epoch_val_loss += loss.item()

            epoch_val_loss /= len(val_loader)

        writer.add_scalars(f'Loss', {'Training': epoch_train_loss, 'Validation': epoch_val_loss}, epoch)

def main():
    parser = argparse.ArgumentParser(description="Training a graph property predictor on the QM9 dataset.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)   

    transform = T.Compose([
        T.NormalizeFeatures(attrs=["y"]),
    ])
    dataset = QM9(root="./data", transform=transform)
    train_dataset, val_dataset = random_split(dataset=dataset, lengths=[0.9, 0.1], generator=torch.manual_seed(69))

    model = GCN(num_node_features=dataset.num_node_features, num_targets=dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    run_training(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        device=device,
        epochs=args.epochs
    )

if __name__ == "__main__":
    main()