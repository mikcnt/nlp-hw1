from typing import Callable
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        num_layers: int,
        hidden_dim: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
        ) -> None:
        super().__init__()

        self.first_layer = nn.Linear(in_features=n_features, out_features=hidden_dim)

        self.layers = (
            nn.ModuleList()
        )

        for i in range(num_layers):
            self.layers.append(
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
            )
            
        self.activation = activation
        
        self.last_layer = nn.Linear(in_features=hidden_dim, out_features=1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, meshgrid: torch.Tensor) -> torch.Tensor:
        out = meshgrid

        out = self.first_layer(
            out
        )  # First linear layer, transforms the hidden dimensions from `n_features` (embedding dimension) to `hidden_dim`
        for layer in self.layers:  # Apply `k` (linear, activation) layer
            out = layer(out)
            out = self.activation(out)
        out = self.last_layer(
            out
        )  # Last linear layer to bring the `hiddem_dim` features to a binary space (`True`/`False`)
        
        out = self.sigmoid(out)
        return out.squeeze(-1)
