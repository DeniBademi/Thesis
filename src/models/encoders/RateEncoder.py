import torch
 
class RateEncoder(torch.nn.Module):
    """ Rate-coding implementation """
    def __init__(self, T):
        super().__init__()
        self.T = T
        
    def forward(self, x):

        # add time dimension after batch dimension
        x = x.unsqueeze(1)
        # repeat x in time dimension
        x = x.repeat(1, self.T, 1, 1, 1)
        
        rands = torch.rand(x.shape, device=x.device)
        spikes = (rands < x).float()

        return tuple([spikes, None, None])
    
    def print_learnable_params(self):
        print("Total Learnable Parameters (encoder):", 0)
