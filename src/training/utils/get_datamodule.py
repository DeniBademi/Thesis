"""Data module factory for creating and configuring data loaders.

This module provides a factory function to create and configure a SpikingDataModule
instance based on configuration parameters. The data module handles dataset loading,
preprocessing, and batch creation for training neural networks.
"""

def get_data_module(config, cv=False):
    """Create and return a configured SpikingDataModule instance.

    This function creates a SpikingDataModule with the specified configuration
    parameters and initializes it for use in training.

    Args:
        config (dict): Configuration dictionary containing data and training specifications.
            Required keys:
            - 'data': Dictionary containing dataset configuration
                - 'dataset': Name of the dataset to use
                - 'data_dir': Directory containing the dataset
            - 'training': Dictionary containing training configuration
                - 'batch_size': Number of samples per training batch

    Returns:
        SpikingDataModule: An initialized data module ready for training.

    Note:
        The data module is automatically set up (initialized) before being returned.
        This includes downloading the dataset if necessary and preparing the data loaders.
    """
    
    data_args = config['data']
    training_args = config['training']
    if cv:
        from src.data.datamodules.SpikingKFoldDataModule import SpikingKFoldDataModule
        data_module = SpikingKFoldDataModule(
            dataset=data_args['dataset'],
            data_dir=data_args['data_dir'],
            batch_size=training_args['batch_size'],
            k=config['training']['k_fold']['k'],
            num_splits=config['training']['k_fold']['num_splits'],
        )
    else:
        from src.data.datamodules.SpikingDataModule import SpikingDataModule
        data_module = SpikingDataModule(
        dataset=data_args['dataset'],
        data_dir=data_args['data_dir'],
        batch_size=training_args['batch_size'],
    )
    data_module.setup()
    return data_module