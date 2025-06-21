import os
import sys
sys.path.append(os.getcwd())
import torch
from pytorch_lightning import Trainer

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from copy import deepcopy
from src.training.tasks import ClassificationTask, EncodingTask
from src.training.utils import (
    get_data_module, 
    get_model, 
    get_loss, 
    get_logger, 
    parse_args, 
    load_config, 
    update_nested_config
)
    
torch.set_float32_matmul_precision('medium')
seed_everything(42)

def wrap_model(model, loss, config):
    backend = 'torch' if config['model']['name'] == "srnn" else 'spikingjelly'
    if config['model']['name'] == 'conv_stal' or config['model']['name'] == 'stal':
        return EncodingTask(model, loss, config['training']['learning_rate'])
    else:
        return ClassificationTask(model, loss, config['training']['learning_rate'], backend=backend)

if __name__ == "__main__":
    
    args = parse_args()
    config = load_config(os.path.join(os.getcwd(), args.config))
    config = update_nested_config(deepcopy(config), vars(args))
    
    data_module = get_data_module(config)
    model = get_model(config)
    loss = get_loss(config)
    
    model = wrap_model(model, loss, config)
    logger = get_logger(config)
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = Trainer(
        max_epochs=config['training']['epochs'],
        logger=logger,
        callbacks=[lr_monitor]
    )
    trainer.fit(model, data_module)
    
    print("="*50+"Training complete"+"="*50)
    if logger is not None:
        logger.finalize('success')
    
    if config['training']['save_weights']:
        from datetime import datetime
        save_dir = "./weights"
        if 'encoder' in config['model']:
            filename = f"{config['model']['name']}_{config['model']['encoder']['name']}_{config['data']['dataset']}"
            if 'freeze_weights' in config['model']['encoder']:
                filename += "_frozen"
            filename += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            filename = f"{config['model']['name']}_{config['data']['dataset']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        trainer.save_checkpoint(os.path.join(save_dir, f'{filename}.cpkt'))
        print(f"Saved checkpoint to {os.path.join(save_dir, f'{filename}.cpkt')}")
    
    print("="*50+"Doing inference on test set"+"="*50)
    test_results = trainer.test(model, data_module)
    print(test_results)