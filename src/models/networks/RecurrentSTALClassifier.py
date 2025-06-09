import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron
from models.networks.ISTAL import STALImageEncoder


from spikingjelly.activation_based import neuron, functional, surrogate, layer, encoding

class RecurrentSTALClassifier(nn.Module):
    def __init__(self, 
                 input_size=32,  # Input image size
                 hidden_channels=64,
                 num_classes=10,
                 num_steps=4,
                 tau=2.0,
                 conv_channels=32,
                 encoder='ISTAL'):
        super().__init__()
        self.num_steps = num_steps
        self.tau = tau
        self.encoder_name = encoder
        # Initialize STAL feature extractor
        
        if encoder == 'ISTAL':
            self.feature_extractor = STALImageEncoder(
                in_channels=1,  # Grayscale images
                conv_channels=conv_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                psi=5,
                alpha=10.0,
                feature_map_size=(input_size, input_size)
            )
            
            # Load pretrained weights
            self.feature_extractor.load_state_dict(torch.load("weights/ISTAL_model.pth"))
        elif encoder == "poisson":
            self.feature_extractor = encoding.PoissonEncoder()
            
            
        # Freeze the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.conv_fc = nn.Sequential(
            layer.Conv2d(1, hidden_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(hidden_channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 14 * 14

            layer.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(hidden_channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 7 * 7

            layer.Linear(hidden_channels * 7 * 7, hidden_channels * 4 * 4, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),

            layer.Linear(hidden_channels * 4 * 4, num_classes, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )
        functional.set_step_mode(self.conv_fc, step_mode='m')
        
    
    def forward(self, x):
        # Extract features using STAL
        b, c, w, h = x.size()

        if self.encoder_name == "ISTAL":
            x, _, _ = self.feature_extractor(x)  # Returns spike_encoding, threshold, potential
        if self.encoder_name == "poisson":
            spikes = torch.full((b, self.num_steps, w, h), 0, dtype=torch.float32, device=x.device)
            for bi in range(b):
                for t in range(self.num_steps):
                    spikes[bi, t] = self.feature_extractor(x[bi,0])
            x = spikes

        x = x.permute(1, 0, 2, 3)
        x = x.unsqueeze(-3)

        x = self.conv_fc(x)
        out = x.mean(0)
        return out
