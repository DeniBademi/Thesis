"""Encoder factory module for creating different types of neural network encoders.

This module provides a factory function to create various types of neural network encoders
based on configuration parameters. It supports different encoder architectures including
STALPS, SPS, Rate, Latency, STAL, and ConvSTAL encoders.
"""

import torch.nn as nn
from ..tasks.encoding_task import EncodingTask
import os

def get_encoder(config: dict) -> nn.Module:
    """Create and return a neural network encoder based on the provided configuration.

    This function acts as a factory for creating different types of neural network encoders.
    The type of encoder and its parameters are specified in the configuration dictionary.

    Args:
        config (dict): Configuration dictionary containing encoder specifications.
            Required keys:
            - 'model': Dictionary containing model configuration
                - 'encoder': Dictionary containing encoder configuration
                    - 'name': Encoder type ('stalps', 'sps', 'rate', 'latency', 'stal', or 'convstal') (case insensitive)
                    - Additional encoder-specific parameters:
                        - For STALPS/SPS:
                            - 'embed_dims': Dimension of the embedding space
                            - 'patch_size': Size of image patches
                        - For Rate/Latency:
                            - 'T': Number of timesteps
                        - For STAL/ConvSTAL:
                            - 'n_spikes_per_timestep': Number of spikes per timestep
                            - 'l1_sz': Size of first layer
                            - 'l2_sz': Size of second layer
                            - 'drop_p': Dropout probability
                            - Optional: 'checkpoint_path': Path to pretrained model checkpoint
                        - For ConvSTAL:
                            - 'kernel_size': Size of convolutional kernel
            - 'data': Dictionary containing dataset information
                - 'dataset': Dataset name ('mnist' or other)

    Returns:
        nn.Module: An instance of the specified neural network encoder.

    Raises:
        ValueError: If the specified encoder type is not supported or if a specified checkpoint path does not exist.

    Supported Encoders:
        - STALPS: Spatio-Temporal Autoencoder with Patch Splitting
        - SPS: Simple Patch Splitting encoder
        - Rate: Rate-based encoding
        - Latency: Latency-based encoding
        - STAL: Spatio-Temporal Autoencoder
        - ConvSTAL: Convolutional Spatio-Temporal Autoencoder
    """
    encoder_type = config['model']['encoder']['name'].lower()
    
    img_dims = (28, 28) if config['data']['dataset'] == 'mnist' else (32, 32)
    in_channels = 1 if config['data']['dataset'] == 'mnist' else 3

    if encoder_type == "stalps":
        from src.models.encoders import STAL_PS
        EMBED_DIMS = config['model']['encoder']['embed_dims']
        patch_size = config['model']['encoder']['patch_size']
        
        encoder = STAL_PS(
            img_height=img_dims[0],
            img_width=img_dims[1],
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=EMBED_DIMS,
            alpha=25
        )
    elif encoder_type == "sps":
        from src.models.encoders import SPS
        EMBED_DIMS = config['model']['encoder']['embed_dims']
        patch_size = config['model']['encoder']['patch_size']

        encoder = SPS(
            img_height=img_dims[0],
            img_width=img_dims[1],
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=EMBED_DIMS
        )
    elif encoder_type == "rate":
        from src.models.encoders import RateEncoder
        T = config['model']['encoder']['T']
        encoder = RateEncoder(T)
    elif encoder_type == "latency":
        from src.models.encoders import LatencyEncoder
        T = config['model']['encoder']['T']
        encoder = LatencyEncoder(T)
    elif encoder_type == "stal":
        
        if 'checkpoint_path' in config['model']['encoder'] :
            
            if os.path.exists(config['model']['encoder']['checkpoint_path']):
                encoder = EncodingTask.load_from_checkpoint(config['model']['encoder']['checkpoint_path']).model
            else:
                raise ValueError(f"Checkpoint path {config['model']['encoder']['checkpoint_path']} does not exist")
        else:
            from src.models.encoders import STAL
            window_size = 28*28 if config['data']['dataset'] == 'mnist' else 32*32
            n_spikes_per_timestep = config['model']['encoder']['n_spikes_per_timestep']
            n_channels = 1 if config['data']['dataset'] == 'mnist' else 3
            l1_sz = config['model']['encoder']['l1_sz']
            l2_sz = config['model']['encoder']['l2_sz']
            drop_p = config['model']['encoder']['drop_p']
        
            encoder = STAL(
                window_size=window_size,
                n_spikes_per_timestep=n_spikes_per_timestep,
                n_channels=n_channels,
                l1_sz=l1_sz,
                l2_sz=l2_sz,
                drop_p=drop_p
            )
    elif encoder_type == "convstal":
        
        if 'checkpoint_path' in config['model']['encoder'] :
            
            if os.path.exists(config['model']['encoder']['checkpoint_path']):
                encoder = EncodingTask.load_from_checkpoint(config['model']['encoder']['checkpoint_path']).model
            else:
                raise ValueError(f"Checkpoint path {config['model']['encoder']['checkpoint_path']} does not exist")
        else:
            from src.models.encoders import ConvSTAL
            n_spikes_per_timestep = config['model']['encoder']['n_spikes_per_timestep']
            kernel_size = config['model']['encoder']['kernel_size']
            drop_p = config['model']['encoder']['drop_p']
            l1_sz = config['model']['encoder']['l1_sz']
  
            encoder = ConvSTAL(
                img_size=img_dims[0],
                n_spikes_per_timestep=n_spikes_per_timestep,
                in_channels=in_channels,
                kernel_size=kernel_size,
                drop_p=drop_p,
                l1_sz=l1_sz,
                l2_sz=in_channels
            )
    else:
        raise ValueError(f"Encoder type {encoder_type} not found")
    return encoder