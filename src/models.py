"""
PRISM Neural Network Architectures:
1. Base Spatial Autoencoder (Geometry -> Latent Space z)
2. Sparse Autoencoder (SAE with L1 Norm Regularization per arXiv:2309.08600)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class SpatialAutoencoder(nn.Module):
    """
    Base Convolutional Autoencoder for PDE Spatial Design Fields rho(x, y).
    Maps 2D density maps (1, H, W) to latent space z in R^{latent_dim} and reconstructs \hat{rho}.
    """
    def __init__(self, in_channels: int = 1, latent_dim: int = 16, img_size: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # Encoder: (1, 32, 32) -> (32, 16, 16) -> (64, 8, 8) -> (128, 4, 4)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Flatten size = 128 * (img_size // 8) * (img_size // 8)
        self.spatial_feat_dim = 128 * (img_size // 8) * (img_size // 8)
        self.fc_enc = nn.Linear(self.spatial_feat_dim, latent_dim)
        
        # Decoder
        self.fc_dec = nn.Linear(latent_dim, self.spatial_feat_dim)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Material density rho bounded in [0, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder: x (B, 1, H, W) -> z (B, latent_dim)."""
        feat = self.encoder_conv(x)
        feat_flat = feat.view(feat.size(0), -1)
        z = self.fc_enc(feat_flat)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder: z (B, latent_dim) -> \hat{x} (B, 1, H, W)."""
        h = self.fc_dec(z)
        feat_spatial = h.view(h.size(0), 128, self.img_size // 8, self.img_size // 8)
        x_hat = self.decoder_conv(feat_spatial)
        return x_hat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder (SAE) with L1 Norm Regularization for Dictionary Learning.
    Reference: Bricken et al., Anthropic 2023 (arXiv:2309.08600)
    
    Given base latent vector z in R^{latent_dim}:
    1. Pre-decoder bias subtraction: z_centered = z - b_d
    2. Feature activation: f = ReLU(W_e * z_centered + b_e)  in R^{num_features}
    3. Reconstructed latent vector: z_hat = W_d * f + b_d  in R^{latent_dim}
    4. Loss: L = ||z - z_hat||^2 / latent_dim + l1_coeff * ||f||_1
    5. Columns of W_d are normalized to ||W_d[:, k]||_2 = 1.
    """
    def __init__(self, latent_dim: int = 16, num_features: int = 64, l1_coeff: float = 1e-2):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_features = num_features
        self.l1_coeff = l1_coeff

        # Decoder bias b_d in R^{latent_dim}
        self.b_d = nn.Parameter(torch.zeros(latent_dim))
        
        # Encoder weights W_e in R^{num_features x latent_dim} and bias b_e in R^{num_features}
        self.W_e = nn.Parameter(torch.empty(num_features, latent_dim))
        self.b_e = nn.Parameter(torch.zeros(num_features))
        
        # Decoder weights W_d in R^{latent_dim x num_features}
        self.W_d = nn.Parameter(torch.empty(latent_dim, num_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with Kaiming / Xavier uniform and unit norm decoder columns."""
        nn.init.kaiming_uniform_(self.W_e, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.W_d, nonlinearity='linear')
        self.normalize_decoder_weights()

    @torch.no_grad()
    def normalize_decoder_weights(self):
        """Normalize columns of decoder matrix W_d to unit L2 norm."""
        norms = torch.norm(self.W_d.data, p=2, dim=0, keepdim=True)
        # Avoid division by zero
        norms = torch.clamp(norms, min=1e-8)
        self.W_d.data = self.W_d.data / norms

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute sparse feature activations f from base latent vector z.
        z shape: (B, latent_dim) -> f shape: (B, num_features)
        """
        z_centered = z - self.b_d.unsqueeze(0)
        f = F.relu(F.linear(z_centered, self.W_e, self.b_e))
        return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct base latent vector z_hat from sparse feature activations f.
        f shape: (B, num_features) -> z_hat shape: (B, latent_dim)
        """
        z_hat = F.linear(f, self.W_d) + self.b_d.unsqueeze(0)
        return z_hat

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass returning (z_hat, f, loss_dict).
        """
        f = self.encode(z)
        z_hat = self.decode(f)

        # 1. Normalized Reconstruction Loss
        recon_loss = F.mse_loss(z_hat, z, reduction='mean')
        
        # 2. L1 Sparsity Penalty on feature activations f (arXiv:2309.08600)
        l1_loss = torch.mean(torch.sum(torch.abs(f), dim=1))
        
        # Total SAE loss
        total_loss = recon_loss + self.l1_coeff * l1_loss

        loss_dict = {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "l1_loss": l1_loss,
            "l0_sparsity": torch.mean((f > 1e-4).float().sum(dim=1))  # Mean active features per sample
        }

        return z_hat, f, loss_dict
