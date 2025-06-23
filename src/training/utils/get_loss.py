"""Loss function factory for creating different types of loss functions.

This module provides a factory function to create various types of loss functions
based on configuration parameters. It supports different loss functions including
CrossEntropy, CE with Spike Sparsity, Encoder Loss, and Multi-Channel Encoder Loss.
"""

def get_loss(config):
    """Create and return a loss function based on the provided configuration.

    This function acts as a factory for creating different types of loss functions
    used in training neural networks. The type of loss function and its parameters
    are specified in the configuration dictionary.

    Args:
        config (dict): Configuration dictionary containing loss function specifications.
            Required keys:
            - 'training': Dictionary containing training configuration
                - 'loss': Dictionary containing loss function configuration
                    - 'name': Loss function type ('cross_entropy', 'ce_ss', 'encoder_loss', or 'multi_channel_encoder_loss')
                    - For 'ce_ss' loss:
                        - 'omega': Weight parameter for spike sparsity
                        - 'c': Weight for concentration loss
                - 'epochs': Number of training epochs (required for 'ce_ss' loss)

    Returns:
        nn.Module: An instance of the specified loss function.

    Raises:
        ValueError: If the specified loss function name is not supported.

    Supported Loss Functions:
        - cross_entropy: Standard Cross Entropy Loss
        - ce_ss: Cross Entropy with Spike Sparsity Loss
        - encoder_loss: Basic Encoder Loss
        - multi_channel_encoder_loss: Multi-Channel Encoder Loss
    """
    loss_args = config['training']['loss']
    if loss_args['name'] == 'cross_entropy':
        from torch.nn import CrossEntropyLoss
        return CrossEntropyLoss()
    elif loss_args['name'] == 'ce_ss':
        from src.training.loss.ce_ss_loss import CE_SpikeSparsityLoss
        omega = loss_args['omega']
        n_epochs = config['training']['epochs']
        c = loss_args['c']
        return CE_SpikeSparsityLoss(omega, n_epochs, c)
    elif loss_args['name'] == 'encoder_loss':
        from src.training.loss.EncoderLoss import EncoderLoss
        return EncoderLoss()
    elif loss_args['name'] == "multi_channel_encoder_loss":
        from src.training.loss.MultiChannelEncoderLoss import MultiChannelEncoderLoss
        return MultiChannelEncoderLoss()
    else:
        raise ValueError(f"Loss function {loss_args['name']} not found")
    