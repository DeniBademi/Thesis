
import torch
from .SpikeSparsityLoss import SpikeSparsityLoss

class CE_SpikeSparsityLoss(torch.nn.Module):
    def __init__(self, omega, n_epochs, concentration_weight):
        super(CE_SpikeSparsityLoss, self).__init__()
        
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.spike_sparsity_loss = SpikeSparsityLoss(omega=omega, n_epochs=n_epochs, concentration_weight=concentration_weight)

    def forward(self, preds, embeddings, epoch_num, y):
        return self.cross_entropy(preds, y) + self.spike_sparsity_loss(embeddings, epoch_num)