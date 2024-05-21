import torch
import torch.nn as nn
import pytorch_lightning as ptl
from pytorch_lightning.loggers import WandbLogger
import numpy as np


def PE(x, degree):
    """
    Positional Encoding
    :param x: value to encode (typically a 3-dimensional coordinate i.e. (x, y, z))
    :param degree: number of frequency bands
    :return: positional encoding of x
    """
    y = torch.cat([2.**i * x for i in range(degree)], dim=-1)
    return torch.cat([x] + [torch.sin(y), torch.cos(y)], dim=-1)


class ResidualINFBlock(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

        self.anatomy_block = [
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU()
        ]

        self.anatomy_block = nn.Sequential(*self.anatomy_block)

    def forward(self, x):
        return x + self.anatomy_block(x)


class DisentangledINF(nn.Module):
    def __init__(self, num_anatomy_blocks=2, num_modality_blocks=2, encoding_dim=256, num_spatial_freq=10, latent_dim=256):
        super().__init__()
        self.anatomy_blocks = num_anatomy_blocks
        self.modality_blocks = num_modality_blocks
        self.W = encoding_dim
        self.num_spatial_freq = num_spatial_freq
        self.latent_dim = latent_dim
        d_xyz = 3 + 6 * num_spatial_freq

        self.xyz_encoding = [
            nn.Linear(d_xyz, latent_dim),
            nn.ReLU()
        ]

        self.anatomy_encoding = [
            nn.Linear(latent_dim + encoding_dim, latent_dim),
            nn.ReLU()
        ]

        self.modality_encoding = [
            nn.Linear(latent_dim + encoding_dim, latent_dim),
            nn.ReLU()
        ]

        self.anatomy_encoder = [
            ResidualINFBlock(latent_dim),
            ResidualINFBlock(latent_dim)
        ]

        # Phi is a scalar value encoding the anatomy at x,y,z - We may ignore it and instead carry the latent
        # code through into the modality encoder - Evaluate experimentally

        # self.phi = nn.Sequential(nn.Linear(latent_dim, 1), nn.Softplus())

        self.modality_encoder = [
            ResidualINFBlock(latent_dim)
        ]

        self.sigma = nn.Sequential(nn.Linear(latent_dim, latent_dim // 2), nn.ReLU(), nn.Linear(latent_dim // 2, 1))

        self.xyz_encoding = nn.Sequential(*self.xyz_encoding)
        self.anatomy_encoding = nn.Sequential(*self.anatomy_encoding)
        self.modality_encoding = nn.Sequential(*self.modality_encoding)
        self.anatomy_encoder = nn.Sequential(*self.anatomy_encoder)
        self.modality_encoder = nn.Sequential(*self.modality_encoder)
        for m in self.modules():
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, coordinates, anatomy_latent, modality_latent):
        """
        Forward pass of the DisentangledINF model
        :param coordinates: 3D coordinates of the input
        :param anatomy_latent: latent code of the anatomy
        :param modality_latent: latent code of the modality
        :return: predicted value of the input
        """
        xyz = PE(coordinates, self.num_spatial_freq)
        x = self.xyz_encoding(xyz)
        x = torch.cat([x, anatomy_latent], dim=-1)
        x = self.anatomy_encoding(x)
        y = self.anatomy_encoder(x)
        # phi = self.phi(y)
        y = torch.cat([y, modality_latent], dim=-1)
        y = self.modality_encoding(y)
        z = self.modality_encoder(y)
        sigma = self.sigma(z)

        return sigma
