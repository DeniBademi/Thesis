"""Hyperparameter optimization module for neural network training.

This module provides functionality for optimizing hyperparameters of neural network
models using Optuna. It specifically focuses on optimizing the omega and concentration
weight parameters for the CE_SpikeSparsityLoss function.
"""

from copy import deepcopy
import wandb
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import optuna
from optuna.trial import Trial

import sys
sys.path.append("./")

from src.training.tasks.classification_task import ClassificationTask
from src.training.loss.ce_ss_loss import CE_SpikeSparsityLoss
from src.training.utils.get_model import get_model
from src.training.utils.get_logger import get_logger
from src.training.utils.get_datamodule import get_data_module
from src.training.utils.args import parse_args, load_config, update_nested_config
from pytorch_lightning import Trainer

def train_with_parameters(config: Dict, omega: float, concentration_weight: float, 
                         num_epochs: int = 5) -> Dict[str, float]:
    """Train a model with specific hyperparameters and return performance metrics.

    This function trains a model using the specified hyperparameters and returns
    the validation accuracy and spike density metrics.

    Args:
        config (Dict): Configuration dictionary containing model and training settings.
        omega (float): Weight parameter for spike sparsity in CE_SpikeSparsityLoss.
        concentration_weight (float): Weight parameter for concentration loss.
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
    config['training']['loss']['concentration_weight'] = concentration_weight
    config['training']['epochs'] = num_epochs
    
    try:
        # Get model and data
        model = get_model(config)
        data_module = get_data_module(config)
        
        # Initialize loss function
        loss_fn = CE_SpikeSparsityLoss(
            omega=omega,
            n_epochs=num_epochs,
            concentration_weight=concentration_weight
        )
        
        # Initialize task
        task = ClassificationTask(
            model=model,
            loss_fn=loss_fn,
            learning_rate=config['training']['learning_rate']
        )
        logger = get_logger(config)
        
        # Train model
        trainer = Trainer(
            max_epochs=config['training']['epochs'],
            logger=logger
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
        - concentration_weight: Float between 0.1 and 10 (log scale)
    """
    # Define hyperparameter search space
    omega = trial.suggest_float('omega', 0.00001, 0.0005, log=True)
    concentration_weight = trial.suggest_float('concentration_weight', 0.1, 10, log=True)
    
    metrics = train_with_parameters(config, omega, concentration_weight, num_epochs)
        
    # Log metrics to Optuna
    trial.set_user_attr('accuracy', metrics['accuracy'])
    trial.set_user_attr('spike_density', metrics['spike_density'])
 
    return metrics['accuracy'], metrics['spike_density']

def find_optimal_parameters(config: Dict, 
                          n_trials: int = 20,
                          num_epochs: int = 5) -> Tuple[float, float]:
    """Find optimal values for omega and concentration weight using Optuna.

    This function performs hyperparameter optimization using Optuna to find the
    best values for omega and concentration weight parameters. It saves the results
    to a file and returns the optimal parameters.

    Args:
        config (Dict): Configuration dictionary containing model and training settings.
        n_trials (int, optional): Number of optimization trials. Defaults to 20.
        num_epochs (int, optional): Number of epochs to train for each trial. Defaults to 5.

    Returns:
        Tuple[float, float]: Optimal values for (omega, concentration_weight).

    Note:
        The optimization results are saved to a file in the directory specified by
        config['training']['logger']['save_dir']/hyperparameter_search/search_results.txt.
        The file includes:
        - Best trial number and score
        - Best parameters (omega and concentration weight)
        - Best trial metrics (accuracy and spike density)
        - Results from all completed trials
    """

    # Create Optuna study
    study = optuna.create_study(
        directions=['maximize', 'minimize'], 
        study_name='ce_ss_loss_optimization',
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, config, num_epochs),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Get best parameters
    best_params = (study.best_params['omega'], study.best_params['concentration_weight'])
    
    # Save results to file
    results_dir = os.path.join(config['training']['logger']['save_dir'], 'hyperparameter_search')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'search_results.txt'), 'w') as f:
        f.write("Optuna Hyperparameter Search Results\n")
        f.write("===================================\n\n")
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
                f.write(f"  Concentration Weight: {trial.params['concentration_weight']:.4f}\n")
                f.write(f"  Accuracy: {trial.user_attrs['accuracy']:.4f}\n")
                f.write(f"  Spike Density: {trial.user_attrs['spike_density']:.4f}\n")

    return best_params

if __name__ == "__main__":
    args = parse_args()
    config = load_config(os.path.join(os.getcwd(), args.config))
    config = update_nested_config(deepcopy(config), vars(args))
    
    optimal_omega, optimal_conc_weight = find_optimal_parameters(config)
    print(f"Optimal parameters found:")
    print(f"Omega: {optimal_omega}")
    print(f"Concentration Weight: {optimal_conc_weight}") 