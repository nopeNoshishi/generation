"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
CnnVae                                   [1, 1, 28, 28]            --
├─Sequential: 1-1                        [1, 512]                  --
│    └─Conv2d: 2-1                       [1, 32, 14, 14]           320
│    └─BatchNorm2d: 2-2                  [1, 32, 14, 14]           64
│    └─ReLU: 2-3                         [1, 32, 14, 14]           --
│    └─Conv2d: 2-4                       [1, 64, 7, 7]             18,496
│    └─BatchNorm2d: 2-5                  [1, 64, 7, 7]             128
│    └─ReLU: 2-6                         [1, 64, 7, 7]             --
│    └─Conv2d: 2-7                       [1, 128, 4, 4]            73,856
│    └─BatchNorm2d: 2-8                  [1, 128, 4, 4]            256
│    └─ReLU: 2-9                         [1, 128, 4, 4]            --
│    └─MaxPool2d: 2-10                   [1, 128, 2, 2]            --
│    └─Flatten: 2-11                     [1, 512]                  --
├─Linear: 1-2                            [1, 2]                    1,026
├─Linear: 1-3                            [1, 2]                    1,026
├─Sequential: 1-4                        [1, 1, 28, 28]            --
│    └─Linear: 2-12                      [1, 512]                  1,536
│    └─ReLU: 2-13                        [1, 512]                  --
│    └─Unflatten: 2-14                   [1, 128, 2, 2]            --
│    └─ConvTranspose2d: 2-15             [1, 128, 4, 4]            262,272
│    └─ConvTranspose2d: 2-16             [1, 64, 7, 7]             73,792
│    └─BatchNorm2d: 2-17                 [1, 64, 7, 7]             128
│    └─ReLU: 2-18                        [1, 64, 7, 7]             --
│    └─ConvTranspose2d: 2-19             [1, 32, 14, 14]           32,800
│    └─BatchNorm2d: 2-20                 [1, 32, 14, 14]           64
│    └─ReLU: 2-21                        [1, 32, 14, 14]           --
│    └─ConvTranspose2d: 2-22             [1, 1, 28, 28]            513
│    └─Sigmoid: 2-23                     [1, 1, 28, 28]            --
==========================================================================================
Total params: 466,277
Trainable params: 466,277
Non-trainable params: 0
Total mult-adds (M): 16.80
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.36
Params size (MB): 1.87
Estimated Total Size (MB): 2.23
==========================================================================================
"""

from typing import TypeVar, Tuple

import torch
from torch import nn

from base_model import BaseModel

Tensor = TypeVar('torch.tensor')
Device = TypeVar('torch.device')

class CnnVae(BaseModel):
    def __init__(self, input_channels: int, z_dim: int, device: Device) -> None:
        super(CnnVae, self).__init__()
        self.z_dim = z_dim
        self.device = device

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding =1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2, 
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2),
            nn.Flatten()
        )

        self.mu_fc = nn.Linear(128*4, z_dim)
        self.log_var_fc = nn.Linear(128*4, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 128*4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 2, 2)),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=input_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
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
        z = torch.randn(1, self.z_dim).to(self.device)
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
