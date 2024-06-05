import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Fully connected residual block.
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int, dropout: float = 0) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, out_features)
        )
        if in_features != out_features:
            self.residual = nn.Linear(in_features, out_features)
        else:
            self.residual = nn.Identity()

        self.batch_norm = nn.BatchNorm1d(out_features)
        self.activation = nn.GELU()

        # zero initialization
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mlp(x)
        y = self.batch_norm(y)
        return self.activation(y + self.residual(x))
