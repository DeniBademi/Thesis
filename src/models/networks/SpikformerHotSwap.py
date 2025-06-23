# Partially based on https://github.com/ZK-Zhou/spikformer

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from .spikformer_modules.block import Block

class SpikformerHotSwap(nn.Module):
    """
    Spikformer with STAL Image Encoding: A Vision Transformer that combines STAL image encoding
    with spiking neurons in the transformer blocks.
    
    This model processes input images through a STAL (Spike Threshold Adaptive Learning) encoder
    to extract features, followed by multiple Transformer blocks with spiking neurons. The model
    operates over multiple time steps (T) and produces class predictions as output.
    
    The model processes input tensors with shape (B, C, H, W) where B is batch size,
    C is channels, H is height, and W is width. The output has shape (B, num_classes).
    
    This implementation provides flexibility to hot swap different image encoders (e.g. SPS, ConvSTAL)
    """
    def __init__(self,encoder, num_classes=10,
                 embed_dims=256, num_heads=4, mlp_ratios=4, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=6, sr_ratios=2, T=4, 
                 # Encoder specific parameters
                  train_encoder=False, encoder_out_channels = 1, img_height=128, img_width=128,
                 ):
        super().__init__()
        self.T = T  # time step
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        
        self.img_height = img_height
        self.img_width = img_width
        self.encoder_out_channels = encoder_out_channels
        

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        # freeze encoder weights
        for param in encoder.parameters():
            param.requires_grad = train_encoder

        # Add a projection layer to convert from conv_channels to embed_dims
        if encoder_out_channels != embed_dims:
            self.projection = nn.Sequential(
                nn.Conv2d(encoder_out_channels, embed_dims, kernel_size=1),
                nn.LayerNorm(embed_dims)
            )
        else:
            self.projection = nn.Identity()

        block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        setattr(self, f"encoder", encoder)
        setattr(self, f"block", block)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore # type: ignore
    def _get_pos_embed(self, pos_embed, encoder, H, W):
        if H * W == self.encoder.num_patches: # type: ignore
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, encoder.H, encoder.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        block = getattr(self, f"block")
        encoder = getattr(self, f"encoder")

        x = encoder(x)  # Shape: (B, conv_channels, H, W)
        
        embeddings = x
        
        # Convert to (T, B, S, C)
        # if isinstance(encoder, ConvSTAL): # Convert (B, in_channels, H, W) to (T, B, S, C)
        if encoder.__class__.__name__ == 'ConvSTAL':
            x = x[0]
            B, T, S = x.shape # batch, time, spikes
            print(x.shape)
            
            x = x.reshape(B, T, self.encoder_out_channels, self.img_height, self.img_width)
            # print("Restore shape: ", x.shape)
            
            # Apply projection to each time step
            x_new = torch.zeros(B, T, self.embed_dims, self.img_height, self.img_width).to(x.device)
            for t in range(T):
                x_new[:,t] = self.projection(x[:,t])
            
            x = x_new
            
            x = x.permute(1, 0, 2, 3, 4) # move time to the front
            x = x.flatten(-2, -1) # flatten the spatial dimensions
            x = x.permute(0, 1, 3, 2) # move channels to last dim
        # print("Encoder output shape: ", x.shape)
        # Process through transformer blocks
        for blk in block:
            x = blk(x)
        
        # Mean over spatial dimension
        x = x.mean(dim=2)  # Shape: (T, B, embed_dims)
        return x, embeddings

    def forward(self, x):
        x, embeddings = self.forward_features(x)
        x = self.head(x.mean(0))  # Mean over time dimension
        return x, embeddings 