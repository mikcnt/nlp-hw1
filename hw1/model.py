from typing import List, Dict

import torch
from torch import nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        n_features = 600
        num_layers = 5
        hidden_dim = 150

        self.first_layer = nn.Linear(in_features=n_features, out_features=hidden_dim)

        self.layers = (
            nn.ModuleList()
        )

        for _ in range(num_layers):
            self.layers.append(
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
            )
        self.activation = torch.nn.functional.relu
        self.last_layer = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, meshgrid: torch.Tensor) -> torch.Tensor:
        out = meshgrid

        out = self.first_layer(
            out
        )
        for layer in self.layers:  # Apply `k` (linear, activation) layer
            out = layer(out)
            out = self.activation(out)
            # out = self.batchnorm(out)
            # out = nn.Dropout(p=0.2)(out)
        out = self.last_layer(
            out
        )  # Last linear layer to bring the `hiddem_dim` features to a binary space (`True`/`False`)
        
        out = self.sigmoid(out)
        return out.squeeze(-1)
