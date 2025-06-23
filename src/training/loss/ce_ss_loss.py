
import torch
from .SpikeSparsityLoss import SpikeSparsityLoss

class CE_SpikeSparsityLoss(torch.nn.Module):
    """The CE+SS loss function for jointly training the encoder and the classifier.

    Args:
        omega (int): The sparsity parameter for the spike sparsity loss.
        n_epochs (int): The number of epochs for the spike sparsity loss.
        c (int): The `c` parameter for the spike sparsity loss.
    """
    def __init__(self, omega, n_epochs, c):
        super(CE_SpikeSparsityLoss, self).__init__()
        
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.spike_sparsity_loss = SpikeSparsityLoss(omega=omega, n_epochs=n_epochs, c=c)

    def forward(self, preds, embeddings, epoch_num, y):
        return self.cross_entropy(preds, y) + self.spike_sparsity_loss(embeddings, epoch_num)