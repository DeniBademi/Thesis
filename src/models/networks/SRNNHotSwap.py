import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron

from .RecurrentClassifier import RecurrentClassifier
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
class SRNNHotSwap(nn.Module):
    def __init__(self, 
                 encoder: nn.Module,
                 window_size: int, 
                 n_spikes_per_timestep: int, 
                 in_channels: int, 
                 l1_sz: int,
                 n_classes: int, 
                 num_steps: int,
                 lif_beta: float = 0.9,
                 freeze_encoder: bool = False):
        super().__init__()
        self.num_classes = 10
        
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen")
        
        self.classifier = RecurrentClassifier(window_size=window_size, 
                                              n_spikes_per_timestep=n_spikes_per_timestep, 
                                              n_channels=in_channels, 
                                              lif_beta=lif_beta,
                                              l1_sz=l1_sz,
                                              n_classes=n_classes,
                                              num_steps=num_steps)

    
    def forward(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                x = self.encoder(x)
        else:
            x = self.encoder(x)
        if isinstance(x, tuple):
            x = x[0]
            
        embeddings = x
        spk_rec, mem_rec = self.classifier(x)
        
        spikes = spk_rec.sum(0)
        return spikes, embeddings