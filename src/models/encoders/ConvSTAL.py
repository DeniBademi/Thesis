import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class RepeatLayer(nn.Module):
    """
    Copies neurons to match the target size.
    The source size must be a divisor of the target size.
    """
    def __init__(self, source_size, target_size):
        super(RepeatLayer, self).__init__()
        assert target_size > source_size, f"target_size must be greater than source_size: source={source_size}, target={target_size}."
        assert target_size % source_size == 0, f"target_size must be a multiple of source_size: source={source_size}, target={target_size}."

        self.repeat_param = target_size // source_size

    def forward(self, x):
        return x.repeat_interleave(self.repeat_param, dim=1)

class ChannelInterpolateLayer(nn.Module):
    """This layer interpolates the input channels to a target number of channels.
    Args:
        size (int): The target channel count.
    """
    def __init__(self, size):
        super(ChannelInterpolateLayer, self).__init__()
        self.size = size

    def forward(self, x):
        B, C_in, H, W = x.shape
        C_out = self.size
        
        x = x.view(B, C_in, H*W)
        x = x.transpose(1, 2)
        x = x.reshape(B, H*W, C_in)
        
        x = torch.nn.functional.interpolate(x, size=(C_out)) 
        
        x = x.reshape(B, H*W, C_out)
        x = x.transpose(1, 2)
        x = x.view(B, C_out, H, W)
        
        return x

class Block(nn.Module):
    def __init__(self, input_channels, in_channels, out_channels, drop_p, kernel_size, name):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(drop_p)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        
        if out_channels == input_channels:
            self.match = torch.nn.Identity()
        elif out_channels > input_channels:
            # project
            self.match = torch.nn.Conv2d(out_channels, input_channels, 1, padding=0, bias=False)
            #instead, avg pool across the channels
            self.match = ChannelInterpolateLayer(input_channels)
            
        elif out_channels < input_channels:
            if input_channels % out_channels != 0:
                raise ValueError(f"{name}: in_channels must be a multiple of out_channels: input_channels={input_channels}, out_channels={out_channels}.")
            self.match = RepeatLayer(out_channels, input_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        match_x = self.match(x)
        match_x = torch.relu(match_x)
        return match_x, x  

class ConvSTAL(nn.Module):
    def __init__(self,
                 img_size:int,
                 n_spikes_per_timestep:int,
                 in_channels:int,
                 kernel_size:int,
                 drop_p:float,
                 l1_sz:int,
                 l2_sz:int,
                 alpha:float=25,
                 flatten_1_dim:bool=False,
                 flatten_hw:bool=False
                 ):
        super().__init__()
        self.__name__ = "ConvSTAL"
        self.img_size = img_size
        self.l1_sz = l1_sz
        self.l2_sz = l2_sz
        self.drop_p = drop_p
        self.alpha = alpha

        self.Z1 = Block(in_channels, in_channels, l1_sz, drop_p, kernel_size, "Z1")
        self.Z2 = Block(in_channels, l1_sz, l2_sz, drop_p, kernel_size, "Z2")
        self.n_spikes_per_timestep = n_spikes_per_timestep
        
        # Learnable threshold parameters
        # Initialzed in the middle such that no large updates are needed (e.g. 0.99 -> 0.01)
        self.threshold_adder = nn.Parameter(torch.Tensor(n_spikes_per_timestep, l2_sz, img_size, img_size).uniform_(0.4, 0.6))

        self.flatten_1_dim = flatten_1_dim
        self.flatten_hw = flatten_hw
    def forward(self, x):
        z1, x = self.Z1(x)
        z2, x = self.Z2(x)
        
        # Expand the feature map along a new dimension corresponding to psi repetitions.
        # x_repeated: (B, psi, conv_channels, H, W)
        x_repeated = x.unsqueeze(1).repeat(1, self.n_spikes_per_timestep, 1, 1, 1) # (B, conv_channels, H, W) -> (B, psi, conv_channels, H, W)

        # The learnable threshold is applied element-wise.
        # Note: self.threshold is (psi, conv_channels, H, W) and is broadcasted to match batch dimension.
        # Apply the surrogate function: sigma(alpha * (h - threshold))
        spikes = torch.sigmoid(self.alpha * (x_repeated - self.threshold_adder)) # encode info spatially
        # Clamp the thresholds to avoid numerical instability
        with torch.no_grad():
            self.threshold_adder.clamp_(0.001, 1.0)

        if self.flatten_1_dim:
            spikes = spikes.flatten(start_dim=-4)
            z1 = z1.flatten(start_dim=-3)
            z2 = z2.flatten(start_dim=-3)
        elif self.flatten_hw:
            spikes = spikes.flatten(start_dim=-3) # flatten H, W and psi
            
            z1 = z1.flatten(start_dim=-2)
            z2 = z2.flatten(start_dim=-2)

        return tuple([spikes, z1, z2])

    def print_learnable_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Total Learnable Parameters (encoder):", total_params)
