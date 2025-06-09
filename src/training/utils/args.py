import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation model')
    parser.add_argument('--config', type=str, default='./config/train.yaml', help='Path to config YAML file')
    # Model args
    parser.add_argument('--model.name', type=str, help='Model architecture')
    parser.add_argument('--model.n_channels', type=int, help='Number of input channels')
    parser.add_argument('--model.n_classes', type=int, help='Number of output classes')
    parser.add_argument('--model.learning_rate', type=float, help='Learning rate')
    # Data args
    parser.add_argument('--data.module', type=str, help='DataModule to use')
    parser.add_argument('--data.data_dir', type=str, help='Data directory')
    parser.add_argument('--data.batch_size', type=int, help='Batch size')
    parser.add_argument('--data.num_workers', type=int, help='Number of data loader workers')
    # Training args
    parser.add_argument('--training.epochs', type=int, help='Number of epochs')
    parser.add_argument('--training.run_name', type=str, help='wandb run name')
    parser.add_argument('--logger.name', type=str, help='logger name')
    parser.add_argument('--training.save_weights', type=bool, help='Save weights')
    parser.add_argument('--training.save_dir', type=str, help='Save directory')
    # Add more as needed
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def update_nested_config(config, flat_args):
    # flat_args: dict with keys like 'model.n_channels'
    for key, value in flat_args.items():
        if value is not None and key != 'config':
            if '.' in key:
                section, subkey = key.split('.', 1)
                if section in config and isinstance(config[section], dict):
                    config[section][subkey] = value
                else:
                    config[section] = {subkey: value}
            else:
                config[key] = value
    return config