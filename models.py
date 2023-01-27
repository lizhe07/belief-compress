import torch.nn as nn

from typing import Optional

NONLINEARITIES = {
    'ReLU': nn.ReLU,
    'ELU': nn.ELU,
}


class MultiLayerPerceptron(nn.Module):

    def __init__(self,
        in_features: int,
        out_features: int,
        num_features: Optional[list[int]] = None,
        nonlinearity: str = 'ReLU',
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_features = num_features or []
        self.nonlinearity = nonlinearity

        num_layers = len(self.num_features)+1
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            _in_features = self.in_features if i==0 else self.num_features[i-1]
            _out_features = self.out_features if i==num_layers-1 else self.num_features[i]
            self.layers.append(nn.Sequential(
                nn.Linear(_in_features, _out_features, bias=True),
                NONLINEARITIES[self.nonlinearity]() if i<num_layers-1 else nn.Identity(),
            ))

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class AutoEncoder(nn.Module):

    def __init__(self,
        in_features: int,
        latent_dim: int,
        num_features: Optional[list[int]] = None,
        **kwargs,
    ):
        super().__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.num_features = num_features or []

        self.encoder = MultiLayerPerceptron(
            in_features, latent_dim, self.num_features, **kwargs,
        )
        self.decoder = MultiLayerPerceptron(
            latent_dim, in_features, list(reversed(self.num_features)), **kwargs,
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y
