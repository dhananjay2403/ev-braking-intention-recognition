'''
    We are implementing a sequence autoencoder that:
        - Takes input: (batch, time_steps=75, features=3)
        - Learns a compressed latent representation per timestep
        - Reconstructs the original input
        - Is trained independently (pretraining)

    Autoencoder design:

        Encoder (per timestep)
        3 → 8 → 4   (latent_dim = 4)

        Decoder
        4 → 8 → 3

    Applied independently at each timestep, but implemented efficiently.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceAutoencoder(nn.Module):

    def __init__(self, input_dim = 3, latent_dim = 4):
        super().__init__()

        # Encoder 
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim)
        )

        # Decoder 
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )


    def forward(self, x):

        # x shape: (batch_size, time_steps, input_dim)
        batch_size, time_steps, input_dim = x.shape         

        # Flatten time dimension for per-step encoding
        x_flat = x.view(batch_size * time_steps, input_dim)

        # Encode
        latent = self.encoder(x_flat)

        # Decode
        reconstructed = self.decoder(latent)

        # Restore original shape
        reconstructed = reconstructed.view(batch_size, time_steps, input_dim)

        return reconstructed


    def encode(self, x):            #  Returns latent representation only.

        batch_size, time_steps, input_dim = x.shape
        x_flat = x.view(batch_size * time_steps, input_dim)
        latent = self.encoder(x_flat)
        latent = latent.view(batch_size, time_steps, -1)

        return latent


