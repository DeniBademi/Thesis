import torch
import numpy as np
import itertools
from .EncoderLoss import EncoderLoss

class MultiChannelEncoderLoss(torch.nn.Module):
    """
    Wrapper for EncoderLoss that handles multi-channel inputs (e.g., RGB images).
    Processes each channel individually and aggregates the results.
    """
    def __init__(self):
        super(MultiChannelEncoderLoss, self).__init__()
        self.__name__ = "MultiChannelEncoderLoss"
        self.encoder_loss = EncoderLoss()
        print(f"{self.__name__} initialized.")

    def forward(self, preds, X):
        """
        W: torch.Tensor[batch, n_spikes_per_timestep, in_channels*height*width], encoded 'spiketrain' for multi-channel input
        X: torch.Tensor[batch, in_channels, height, width] or torch.Tensor[batch, in_channels, height*width], multi-channel input signal (e.g., RGB image)
        Z1: torch.Tensor[batch, in_channels, height*width], representation of Layer 1
        Z2: torch.Tensor[batch, in_channels, height*width], representation of Layer 2
        """
        W, Z1, Z2 = preds
        assert isinstance(X, torch.Tensor) and isinstance(W, torch.Tensor), "X and W must be torch tensors."
        assert X.ndim >= 3, f"X must have at least 3 dimensions [batch, channels, omega*psi], got {X.ndim}"

        # Flatten the input if it has 4 dimensions
        if X.ndim == 4:
            X = X.flatten(start_dim=-2)
    
        B, C, _ = X.shape
        _, n_spikes_per_timestep, _ = W.shape
        
        # Prepare to store individual channel losses
        channel_losses = []
        
        # Process each channel individually
        for c in range(C):
            X_c = X[:, c] if X.ndim == 3 else X  # [batch, omega*psi] for this channel
            
            # Extract channel-specific data for W, Z1, Z2
            W_unflat = W.reshape(B, n_spikes_per_timestep, C, -1) # [batch, n_spikes, n_channels, -1]
            W_c = W_unflat[:, :, c] if W.ndim == 3 else W  # Handle either [batch, channels, omega*psi] or [batch, omega*psi]
            W_c = W_c.reshape(B, -1)
            Z1_c = None
            if Z1 is not None:
                Z1_c = Z1[:, c] if Z1.ndim == 3 else Z1
                
            Z2_c = None
            if Z2 is not None:
                Z2_c = Z2[:, c] if Z2.ndim == 3 else Z2
            
            # Create channel-specific prediction tuple
            preds_c = (W_c, Z1_c, Z2_c)
            
            # Compute loss for this channel using the original EncoderLoss
            channel_loss = self.encoder_loss(preds_c, X_c)
            channel_losses.append(channel_loss)
        
        # Average the losses across all channels
        total_loss = torch.mean(torch.stack(channel_losses))
        return total_loss
    