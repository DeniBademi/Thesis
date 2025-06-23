# Inspired from "Explicitly Trained Spiking Sparsity in Spiking
# Neural Networks with Backpropagation" by Jason M. Allred, Steven J. Spencer,
# Gopalakrishnan Srinivasan, and Kaushik Roy.

import torch
import torch.nn as nn

class SpikeSparsityLoss(nn.Module):
    """
    The spike sparsity loss function that gradually increases the penalty for dense spike features.
    
    Args:
        omega (int): The sparsity parameter for the spike sparsity loss.
        n_epochs (int): The number of epochs for the spike sparsity loss.
        c (int): Scaling factor for the L2/L1 ratio term.
    """
    
    def __init__(self, omega, n_epochs, c):
        super(SpikeSparsityLoss, self).__init__()
        self.omega = omega
        self.n_epochs = n_epochs
        self.c = c

    def forward(self, embeddings, epoch_num):
        #input shape (T, B, N, C)
        # Compute spike count for each embedding
        
        # Compute mean spike count across all embeddings in the batch
        spike_count = torch.mean(torch.sum(embeddings, dim=0), dim=0)
        
        # Calculate L1 penalty on total activation
        l1_penalty = torch.sum(spike_count)

        # Calculate local importance or concentration measure
        concentration = torch.sum(spike_count**2) / (torch.sum(spike_count) + 1e-6)

        # Loss that reduces total activity but rewards concentration
        loss = self.omega * (epoch_num / self.n_epochs) * (l1_penalty - self.c * concentration)
        
        return loss
        
