import torch
from spikingjelly.activation_based.encoding import LatencyEncoder as SNNLatencyEncoder

class LatencyEncoder(torch.nn.Module):
    """ Latency-coding implementation """
    def __init__(self, T):        
        super().__init__()
        self.T = T
        self.encoder = SNNLatencyEncoder(T=T, step_mode="m")

    def forward(self, x):
        
        # add time dimension after batch dimension
        x = x.unsqueeze(1)
        # repeat x in time dimension
        x = x.repeat(1, self.T, 1, 1, 1)
        # encode
        spikes = self.encoder(x)

        return tuple([spikes, None, None])
    
    def print_learnable_params(self):
        print("Total Learnable Parameters (encoder):", 0)
