"""
Neural Network Architectures for Hybrid Medical Diagnostics PACS Suite.
Includes ConvAE Anomaly Detection Model and Deep Representation MLP Head.
"""

import torch
import torch.nn as nn


class ConvAE(nn.Module):
    """
    Convolutional Autoencoder with latent bottleneck for normative chest radiograph reconstruction
    and spatial anomaly residual isolation.
    """
    def __init__(self, latent: int = 128, img: int = 128):
        super().__init__()
        s = img // 16

        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 4, 2, 1),
                nn.BatchNorm2d(o),
                nn.LeakyReLU(0.2)
            )

        def dblk(i, o):
            return nn.Sequential(
                nn.ConvTranspose2d(i, o, 4, 2, 1),
                nn.BatchNorm2d(o),
                nn.LeakyReLU(0.2)
            )

        self.s = s
        self.enc = nn.Sequential(
            blk(1, 32),
            blk(32, 64),
            blk(64, 128),
            blk(128, 256),
            nn.Flatten(),
            nn.Linear(256 * s * s, latent)
        )
        self.fc = nn.Linear(latent, 256 * s * s)
        self.dec = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            dblk(256, 128),
            dblk(128, 64),
            dblk(64, 32),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        z = self.enc(x)
        rec = self.dec(self.fc(z).view(-1, 256, self.s, self.s))
        return rec, z


def build_tuned_mlp_head() -> nn.Sequential:
    """
    Builds the deep bottleneck representation MLP head matching trained checkpoint weights.
    Structure: Linear(1024->256) -> ReLU -> Dropout(0.3) -> Linear(256->128) -> ReLU -> Dropout(0.3) -> Linear(128->1).
    """
    return nn.Sequential(
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1)
    )
