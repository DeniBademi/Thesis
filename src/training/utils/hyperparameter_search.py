"""Hyperparameter optimization module for neural network training.

This module provides functionality for optimizing hyperparameters of neural network
models using Optuna. It specifically focuses on optimizing the omega and concentration
weight parameters for the CE_SpikeSparsityLoss function.
"""

from copy import deepcopy
import wandb
import os
from typing import Dict
import optuna
from optuna.trial import Trial
from datetime import datetime
import torch
import sys
sys.path.append("./")

from src.training.tasks.classification_task import ClassificationTask
from src.training.loss.ce_ss_loss import CE_SpikeSparsityLoss
from src.training.utils.get_model import get_model
from src.training.utils.get_logger import get_logger
from src.training.utils.get_datamodule import get_data_module
from src.training.utils.args import parse_args, load_config, update_nested_config
from pytorch_lightning import Trainer, seed_everything

torch.set_float32_matmul_precision('medium')
seed_everything(42)

def train_with_parameters(config: Dict, omega: float, c: float, 
                         num_epochs: int = 5) -> Dict[str, float]:
    """Train a model with specific hyperparameters and return performance metrics.

    This function trains a model using the specified hyperparameters and returns
    the validation accuracy and spike density metrics.

    Args:
        config (Dict): Configuration dictionary containing model and training settings.
        omega (float): Weight parameter for spike sparsity in CE_SpikeSparsityLoss.
        c (float): Weight parameter for concentration loss.
        num_epochs (int, optional): Number of training epochs. Defaults to 5.

    Returns:
        Dict[str, float]: Dictionary containing validation metrics:
            - 'accuracy': Validation accuracy
            - 'spike_density': Validation spike density

    Raises:
        ValueError: If the logger is not available or properly configured.
        Exception: Any other exception that occurs during training.
    """
    # Update config with current parameters
    config['training']['loss']['omega'] = omega
    config['training']['loss']['c'] = c
    config['training']['epochs'] = num_epochs
    
    try:
        # Get model and data
        model = get_model(config)
        data_module = get_data_module(config)
        
        # Initialize loss function
        loss_fn = CE_SpikeSparsityLoss(
            omega=omega,
            n_epochs=num_epochs,
            c=c
        )
        
        # Initialize task
        task = ClassificationTask(
            model=model,
            loss_fn=loss_fn,
            learning_rate=config['training']['learning_rate'],
            backend= 'spikingjelly' if config['model']['name'] == 'spikformer' else 'pytorch'
        )
        logger = get_logger(config)
        
        # Train model
        trainer = Trainer(
            max_epochs=config['training']['epochs'],
            logger=logger,
        )
        trainer.fit(task, data_module)
        
        # Get metrics from the trainer's logger
        if logger is not None and hasattr(logger, 'experiment'):
            metrics = {
                'accuracy': logger.experiment.summary.get('val_accuracy_epoch', 0.0),
                'spike_density': logger.experiment.summary.get('val_spike_density_epoch', 0.0)
            }
            print(metrics)
        else:
            raise ValueError("Logger is not available")
        
        return metrics
    except Exception as e:
        raise e
    finally:
        wandb.finish()

def objective(trial: Trial, config: Dict, num_epochs: int):
    """Objective function for Optuna optimization.

    This function defines the hyperparameter search space and evaluates a single
    trial by training the model with the suggested parameters.

    Args:
        trial (Trial): Optuna trial object for hyperparameter optimization.
        config (Dict): Configuration dictionary containing model and training settings.
        num_epochs (int): Number of epochs to train for each trial.

    Returns:
        float: Combined score for optimization (4 * accuracy - spike_density).

    Note:
        The hyperparameter search space is defined as:
        - omega: Float between 0.00001 and 0.0005 (log scale)
        - c: Float between 0.1 and 10 (log scale)
    """
    # Define hyperparameter search space
    # omega = trial.suggest_float('omega', 0.000001, 0.0005, log=True)
    omega = trial.suggest_float('omega', 0.00026, 0.0005, log=True)
    
    c = trial.suggest_float('c', 0.1, 10, log=True)
    
    metrics = train_with_parameters(config, omega, c, num_epochs)
        
    # Log metrics to Optuna
    trial.set_user_attr('accuracy', metrics['accuracy'])
    trial.set_user_attr('spike_density', metrics['spike_density'])
 
    return metrics['accuracy'], metrics['spike_density']

def save_search_results(study: optuna.Study, config: Dict):
    
    log_filename = os.path.join(config['training']['logger']['save_dir'], \
        'hyperparameter_search', f'search_results_{config["model"]["name"]}_{config["data"]["dataset"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    
    with open(log_filename, 'w') as f:
        f.write("Optuna Hyperparameter Search Results\n")
        f.write("===================================\n\n")
    
        if study.directions:
            f.write(f"Number of trials on the Pareto front: {len(study.best_trials)}\n")
        
            trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[0])
            f.write("Trial with highest accuracy: ")
            f.write(f"\tnumber: {trial_with_highest_accuracy.number}\n")
            f.write(f"\tparams: {trial_with_highest_accuracy.params}\n")
            f.write(f"\tvalues: {trial_with_highest_accuracy.values}\n")
            
            trial_with_lowest_spike_density = min(study.best_trials, key=lambda t: t.values[1])
            f.write("Trial with lowest spike density: ")
            f.write(f"\tnumber: {trial_with_lowest_spike_density.number}\n")
            f.write(f"\tparams: {trial_with_lowest_spike_density.params}\n")
            f.write(f"\tvalues: {trial_with_lowest_spike_density.values}\n")
        else:
            best_params = (study.best_params['omega'], study.best_params['c'])
            
            f.write(f"Best trial: {study.best_trial.number}\n")
            f.write(f"Best score: {study.best_value:.4f}\n")
            f.write(f"Best parameters:\n")
            f.write(f"  Omega: {best_params[0]:.6f}\n")
            f.write(f"  Concentration Weight: {best_params[1]:.4f}\n\n")
                
            f.write("Best trial metrics:\n")
            f.write(f"  Accuracy: {study.best_trial.user_attrs['accuracy']:.4f}\n")
            f.write(f"  Spike Density: {study.best_trial.user_attrs['spike_density']:.4f}\n")

            f.write("All trials:\n")
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    f.write(f"\nTrial {trial.number}:\n")
                    f.write(f"  Score: {trial.value:.4f}\n")
                    f.write(f"  Omega: {trial.params['omega']:.6f}\n")
                    f.write(f"  Concentration Weight: {trial.params['c']:.4f}\n")
                    f.write(f"  Accuracy: {trial.user_attrs['accuracy']:.4f}\n")
                    f.write(f"  Spike Density: {trial.user_attrs['spike_density']:.4f}\n")
            

def find_optimal_parameters(config: Dict):
    # study = optuna.create_study(
    #     storage=f'sqlite:///{os.path.join(config["training"]["logger"]["save_dir"], "hyperparameter_search", f"study_{config["model"]["name"]}_{config["data"]["dataset"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db")}',
    #     directions=['maximize', 'minimize'], 
    #     study_name='ce_ss_loss_optimization',
    #     pruner=optuna.pruners.MedianPruner()
    # )
    
    study = optuna.load_study(
        study_name='ce_ss_loss_optimization',
        storage='sqlite:///C:/Users/dzahariev/Desktop/Thesis/Thesis/experiments/hyperparameter_search/study_srnn_mnist_20250609_205015.db',
        pruner=optuna.pruners.MedianPruner()
    )
    n_trials = config['training']['n_trials']
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, config, config['training']['epochs']),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    save_search_results(study, config)

if __name__ == "__main__":
    args = parse_args()
    config = load_config(os.path.join(os.getcwd(), args.config))
    config = update_nested_config(deepcopy(config), vars(args))
    
    find_optimal_parameters(config)