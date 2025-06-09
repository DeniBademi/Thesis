"""Model factory module for creating different types of neural network models.

This module provides a factory function to create various types of neural network models
based on configuration parameters. It supports different model architectures including
Spikformer, SRNN, ConvSTAL, and STAL models.
"""

import torch.nn as nn
from .get_encoder import get_encoder

def get_model(config: dict) -> nn.Module:
    """Create and return a neural network model based on the provided configuration.

    This function acts as a factory for creating different types of neural network models.
    The type of model and its parameters are specified in the configuration dictionary.

    Args:
        config (dict): Configuration dictionary containing model specifications.
            Required keys:
            - 'model': Dictionary containing model configuration
                - 'name': Model type ('spikformer', 'srnn', 'conv_stal', or 'stal') (case sensitive)
                - Additional model-specific parameters
            - 'data': Dictionary containing dataset information
                - 'dataset': Dataset name ('mnist' or other)

    Returns:
        nn.Module: An instance of the specified neural network model.

    Raises:
        ValueError: If the specified model name is not supported.

    Supported Models:
        - Spikformer: A transformer-based model with spike encoding
        - SRNN: Spiking Recurrent Neural Network
        - ConvSTAL: Convolutional Spatio-Temporal Autoencoder
        - STAL: Spatio-Temporal Autoencoder
    """
    model_args = config['model']
    if model_args['name'] == 'spikformer':
        encoder = get_encoder(config)
        
        from src.models.networks import SpikformerHotSwap
        model = SpikformerHotSwap(
            encoder=encoder,
            embed_dims=config['model']['encoder']['embed_dims'],
            num_heads=config['model']['num_heads'],
            mlp_ratios=config['model']['mlp_ratios'],
            depths=config['model']['depths'],
            sr_ratios=config['model']['sr_ratios'],
            train_encoder=True,
            encoder_out_channels=config['model']['encoder']['embed_dims'],
            img_height= 28 if config['data']['dataset'] == 'mnist' else 32,
            img_width= 28 if config['data']['dataset'] == 'mnist' else 32,
        )
    elif model_args['name'] == 'srnn':
        from src.models.networks import SRNNHotSwap
        
        encoder = get_encoder(config)
        
        model = SRNNHotSwap(
            encoder=encoder,
            window_size= 28*28 if config['data']['dataset'] == 'mnist' else 32*32,
            n_spikes_per_timestep=model_args['n_spikes_per_timestep'],
            in_channels= 1 if config['data']['dataset'] == 'mnist' else 3,
            lif_beta=model_args['lif_beta'],
            l1_sz=model_args['l1_sz'],
            n_classes= 10,
            num_steps=model_args['num_steps'],
            freeze_encoder=bool(model_args['encoder']['freeze_weights']) if 'freeze_weights' in model_args['encoder'] else False
        )
    elif model_args['name'] == 'conv_stal':
        from src.models.encoders import ConvSTAL
        
        model = ConvSTAL(
            img_size= 28 if config['data']['dataset'] == 'mnist' else 32,
            n_spikes_per_timestep=model_args['n_spikes_per_timestep'],
            in_channels= 1 if config['data']['dataset'] == 'mnist' else 3,
            kernel_size=model_args['kernel_size'],
            drop_p=model_args['drop_p'],
            l1_sz=model_args['l1_sz'],
            l2_sz=1 if config['data']['dataset'] == 'mnist' else 3,
            flatten_hw=True if config['training']['loss']['name'] == 'multi_channel_encoder_loss' else False
        )
    elif model_args['name'] == 'stal':
        from src.models.encoders import STAL
        
        model = STAL(
            window_size= 28*28 if config['data']['dataset'] == 'mnist' else 32*32,
            n_channels= 1 if config['data']['dataset'] == 'mnist' else 3,
            l1_sz=model_args['l1_sz'],
            l2_sz=1 if config['data']['dataset'] == 'mnist' else 3,
            drop_p=model_args['drop_p'],
            n_spikes_per_timestep=model_args['n_spikes_per_timestep'],
        )
    else:
        raise ValueError(f"Model {model_args['name']} not found")
    return model