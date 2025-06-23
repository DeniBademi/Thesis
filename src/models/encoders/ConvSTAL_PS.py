import torch
import torch.nn as nn
from timm.models.layers import to_2tuple

class ConvSTAL_PS(nn.Module):
    def __init__(self, img_height=128, img_width=128, patch_size=4, in_channels=2, embed_dims=256, alpha:float=25, T:int=5):
        super().__init__()
        self.image_size = [img_height, img_width]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1] # type: ignore
        self.num_patches = self.H * self.W
        self.alpha = alpha
        self.patch_count = (self.H) * (self.W)
        self.T = T
        
        self.relu = torch.nn.ReLU()

        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)
        # self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.proj_conv1 = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims//4)
        # self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims//2)
        # self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        # self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        # self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.threshold_adder = nn.Parameter(torch.Tensor(self.T, self.patch_count, embed_dims).uniform_(0.4, 0.6))
    def forward(self, x):

        if x.ndim == 4: # time dimension is not present, so repeat it 5 times
            x = x.unsqueeze(0)#.repeat(5, 1, 1, 1, 1)

        T, B, C, H, W = x.shape

        x = x.flatten(0, 1) # flatten time and batch dimensions
        x = self.proj_conv(x) # run convolution on all samples and timesteps
        x = self.proj_bn(x)#.reshape(T, B, -1, H, W).flatten(0, 1)
        x = self.relu(x)

        x = self.proj_conv1(x)
        x = self.proj_bn1(x)#.reshape(T, B, -1, H, W).flatten(0, 1)
        x = self.relu(x)

        x = self.proj_conv2(x)
        x = self.proj_bn2(x)#.reshape(T, B, -1, H, W).flatten(0, 1)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x)#.reshape(T, B, -1, H//2, W//2).flatten(0, 1)
        x = self.relu(x)
        x = self.maxpool3(x)
        
        x_feat = x.reshape(T, B, -1, H//4, W//4)
        
        x = self.rpe_conv(x)
        x = self.rpe_bn(x).reshape(T, B, -1, H//4, W//4)
        x = self.relu(x)
        x = x + x_feat # here x is shape (T, B, embed_dims, H//4, W//4)
        
        x = x.repeat(self.T, 1, 1, 1, 1)
        
        x = x.flatten(-2).transpose(-1, -2).permute(1, 0, 2, 3)  # T, B, C, H//4, W//4 --flatten-> T, B, C, H//4*W//4 --transpose-> T, B, H//4*W//4, C --permute-> B, T, N, C

        for b in range(B):
            x[b] = torch.sigmoid(self.alpha * (x[b] - self.threshold_adder))
        x = torch.sigmoid(self.alpha * (x - self.threshold_adder))
        x = x.permute(1, 0, 2, 3) # T, B, N, C
        
        return x