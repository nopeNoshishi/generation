""" model architecture (Feed Forward Variational Autoencoder)
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
FeedForwardVae                           [100, 1, 784]             --
├─Sequential: 1-1                        [100, 1, 256]             --
│    └─Linear: 2-1                       [100, 1, 512]             401,920
│    └─ReLU: 2-2                         [100, 1, 512]             --
│    └─Linear: 2-3                       [100, 1, 512]             262,656
│    └─ReLU: 2-4                         [100, 1, 512]             --
│    └─Linear: 2-5                       [100, 1, 256]             131,328
│    └─ReLU: 2-6                         [100, 1, 256]             --
├─Linear: 1-2                            [100, 1, 2]               514
├─Linear: 1-3                            [100, 1, 2]               514
├─Sequential: 1-4                        [100, 1, 784]             --
│    └─Linear: 2-7                       [100, 1, 256]             768
│    └─ReLU: 2-8                         [100, 1, 256]             --
│    └─Linear: 2-9                       [100, 1, 512]             131,584
│    └─ReLU: 2-10                        [100, 1, 512]             --
│    └─Linear: 2-11                      [100, 1, 512]             262,656
│    └─ReLU: 2-12                        [100, 1, 512]             --
│    └─Linear: 2-13                      [100, 1, 784]             402,192
│    └─Sigmoid: 2-14                     [100, 1, 784]             --
==========================================================================================
Total params: 1,594,132
Trainable params: 1,594,132
Non-trainable params: 0
Total mult-adds (M): 159.41
==========================================================================================
Input size (MB): 0.31
Forward/backward pass size (MB): 2.68
Params size (MB): 6.38
Estimated Total Size (MB): 9.37
==========================================================================================
"""

from typing import TypeVar, Tuple

import torch
from torch import nn

from base_model import BaseModel

Tensor = TypeVar('torch.tensor')
Device = TypeVar('torch.device')


class FeedForwardVae(BaseModel):
    def __init__(self, input_size: int, z_dim: int, device: Device) -> None:
        super(FeedForwardVae, self).__init__()
        self.z_dim = z_dim
        self.device = device

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.mu_fc = nn.Linear(256, z_dim)
        self.log_var_fc = nn.Linear(256, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_size),
            nn.Sigmoid()
        )

    def encode(self, input: Tensor) -> Tuple[Tensor]:
        x = self.encoder(input)
        mu = self.mu_fc(x)
        log_var = self.log_var_fc(x)

        return mu, log_var

    def decode(self, input: Tensor) -> Tensor:
        return self.decoder(input)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var, z

    def generate(self):
        z = torch.randn(self.z_dim).to(self.device)
        return self.decode(z)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        eps = torch.rand_like(torch.exp(log_var))
        return mu + torch.exp(log_var / 2) * eps

    def loss_function(self, x_re, x, mu, log_var):
        # Reconstruction
        reconstruction_loss = nn.functional.binary_cross_entropy(x_re, x, reduction='sum')
        # Kullback–Leibler divergence
        kl_loss = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # all
        vae_loss = reconstruction_loss + kl_loss

        return vae_loss, reconstruction_loss, kl_loss
