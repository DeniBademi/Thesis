import os
import sys
sys.path.append(os.getcwd())
from sklearn.model_selection import KFold
import numpy as np
from datetime import datetime
from copy import deepcopy
from src.data.datamodules.SpikingDataModule import SpikingDataModule
from src.training.tasks import EncodingTask, ClassificationTask
from src.training.utils.args import parse_args, load_config, update_nested_config
from src.training.utils import get_model, get_loss, get_logger
from pytorch_lightning import Trainer, seed_everything, LightningDataModule, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor
seed_everything(42)

def wrap_model(model, loss, config):
    backend = 'torch' if config['model']['name'] == "srnn" else 'spikingjelly'
    if config['model']['name'] == 'conv_stal' or config['model']['name'] == 'stal':
        return EncodingTask(model, loss, config['training']['learning_rate'])
    else:
        return ClassificationTask(model, loss, config['training']['learning_rate'], backend=backend)

def train_fold(model:LightningModule, data_module:LightningDataModule, config, fold_idx, k_folds=5):
    """Train a single fold of k-fold cross validation."""
    # Create fold-specific logger
    fold_config = deepcopy(config)
    if fold_config['training']['logger']['name'] == 'wandb':
        fold_config['training']['logger']['run_name'] = f"{fold_config['training']['logger'].get('run_name', '')}_fold_{fold_idx}"
    
    logger = get_logger(fold_config)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = Trainer(
        max_epochs=config['training']['epochs'],
        logger=logger,
        callbacks=[lr_monitor]
    )
    
    # Train and test the model
    trainer.fit(model, data_module)
    results = trainer.validate(model, data_module)
    
    if logger is not None:
        logger.finalize('success')
    
    # Save fold-specific checkpoint
    if config['training']['save_weights']:
        save_dir = "C:/Users/dzahariev/Desktop/Thesis/Thesis/weights"
        if 'encoder' in config['model']:
            filename = f"{config['model']['name']}_{config['model']['encoder']['name']}_{config['data']['dataset']}_fold_{fold_idx}"
            if 'freeze_weights' in config['model']['encoder']:
                filename += "_frozen"
            filename += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            filename = f"{config['model']['name']}_{config['data']['dataset']}_fold_{fold_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        trainer.save_checkpoint(os.path.join(save_dir, f'{filename}.cpkt'))
        print(f"Saved fold {fold_idx} checkpoint to {os.path.join(save_dir, f'{filename}.cpkt')}")
    
    return results

def main():
    # Parse arguments and load config
    args = parse_args()
    config = load_config(os.path.join(os.getcwd(), args.config))
    config = update_nested_config(deepcopy(config), vars(args))
    
    # Initialize lists to store results
    all_fold_results = []
    
    # Perform k-fold cross validation
    k_folds = 1
    for fold_idx in range(k_folds):
        print(f"\nTraining fold {fold_idx + 1}/{k_folds}")
        
        # Create new model instance for each fold
        model = get_model(config)
        loss = get_loss(config)
        model = wrap_model(model, loss, config)
        
        data_module = SpikingDataModule(
            dataset=config['data']['dataset'],
            data_dir=config['data']['data_dir'],
            batch_size=config['training']['batch_size'],
            k=fold_idx+1,
            num_splits=k_folds)
        data_module.setup()
        print(f"Length of train dataset: {len(data_module.train_dataset)}")
        print(f"Length of val dataset: {len(data_module.val_dataset)}")
        
        # Train and evaluate the fold
        fold_results = train_fold(model, data_module, config, fold_idx, k_folds)
        all_fold_results.append(fold_results)
        
        # Print fold results
        print(f"\nFold {fold_idx + 1} Results:")
        for metric_name, value in fold_results[0].items():
            print(f"{metric_name}: {value:.4f}")
    
    # Calculate and print average results across all folds
    print("\nCross-validation Results Summary:")
    avg_results = {}
    for metric_name in all_fold_results[0][0].keys():
        values = [fold_result[0][metric_name] for fold_result in all_fold_results]
        avg_results[metric_name] = np.mean(values)
        std_results = np.std(values)
        print(f"{metric_name}: {avg_results[metric_name]:.4f} ± {std_results:.4f}")

if __name__ == "__main__":
    main()