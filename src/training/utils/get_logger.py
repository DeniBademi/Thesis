"""Logger factory for creating and configuring experiment logging.

This module provides a factory function to create and configure experiment loggers
based on configuration parameters. It supports different logging backends including
Weights & Biases (wandb) and MLFlow for tracking experiments.
"""

from pytorch_lightning.loggers import WandbLogger, MLFlowLogger

def get_logger(config) -> WandbLogger | MLFlowLogger | None:
    """Create and return a configured experiment logger.

    This function creates a logger instance based on the specified configuration
    parameters and initializes it with the appropriate hyperparameters.

    Args:
        config (dict): Configuration dictionary containing logger specifications.
            Required keys:
            - 'training': Dictionary containing training configuration
                - 'logger': Dictionary containing logger configuration
                    - 'name': Logger type ('wandb' or 'mlflow') (case insensitive)
                    - 'save_dir': Directory to save logs
                    - 'project': Project name for the experiment
                    - Optional for wandb:
                        - 'offline': Boolean to run wandb in offline mode
            - 'model': Dictionary containing model configuration
                - 'name': Model name
                - Optional 'encoder': Dictionary containing encoder configuration
                    - 'name': Encoder name
                    - Optional 'freeze_weights': Boolean for weight freezing
            - 'data': Dictionary containing dataset information
                - 'dataset': Dataset name
            - 'training': Additional training parameters
                - 'batch_size': Batch size
                - 'learning_rate': Learning rate
                - 'loss': Dictionary containing loss configuration
                    - 'name': Loss function name
                - 'epochs': Number of training epochs

    Returns:
        Union[WandbLogger, MLFlowLogger, None]: An initialized logger instance or None if no logger is configured.

    Raises:
        ValueError: If the specified logger name is not supported.

    Note:
        The run name is automatically generated based on the model, encoder (if present),
        and dataset configuration. For wandb, an offline mode option is available.
        All relevant hyperparameters are automatically logged to the chosen backend.
    """
    
    if 'logger' not in config['training']:
        return None
    
    logger_args = config['training']['logger']
    save_dir = logger_args['save_dir']
    project_name = logger_args['project']
    
    if 'encoder' in config['model']:
        run_name = f"{config['model']['name']}_{config['model']['encoder']['name']}_{config['data']['dataset']}"

        if 'freeze_weights' in config['model']['encoder']:
            suffix = '_frozen' if config['model']['encoder']['freeze_weights'] else ''
            run_name = f"{run_name}{suffix}"
    else:
        run_name = f"{config['model']['name']}_{config['data']['dataset']}"
        
    # add loss fn name
    run_name = f"{run_name}_{config['training']['loss']['name']}"
        
    if logger_args['name'].lower() == 'wandb':
        from pytorch_lightning.loggers import WandbLogger
        is_offline = bool(logger_args['offline']) if 'offline' in logger_args else False
        logger = WandbLogger(name=run_name, project=project_name, save_dir=save_dir, offline=is_offline)
    elif logger_args['name'].lower() == 'mlflow':
        from pytorch_lightning.loggers import MLFlowLogger
        logger = MLFlowLogger(experiment_name=project_name, save_dir=save_dir)
    else:
        raise ValueError(f"Logger {logger_args['name']} not found")
    
    hyperparameters = {
        'dataset': config['data']['dataset'],
        'model': config['model']['name'],
        **{f"mdodel.{k}": v for k, v in config['model'].items() if k != 'name'},
        'batch_size': config['training']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'loss': config['training']['loss']['name'],
        **{f"loss.{k}": v for k, v in config['training']['loss'].items() if k != 'name'},
        'epochs': config['training']['epochs'],
    }
    
    logger.log_hyperparams(hyperparameters)
    
    return logger
