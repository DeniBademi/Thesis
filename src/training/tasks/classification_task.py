
from spikingjelly.clock_driven import functional as SF

from .base_task import BaseTask
from ..loss.CE_L1Loss import CE_L1Loss
from .MetricManager import MetricManager

class ClassificationTask(BaseTask):

    def __init__(self, model, loss_fn, learning_rate,  backend='spikingjelly'):
        super(ClassificationTask, self).__init__(model, loss_fn, learning_rate)
        self.save_hyperparameters("model", "loss_fn", "learning_rate")

        self.backend = backend
        
        self.metric_manager = MetricManager(log_fn = self.log, num_classes=self.num_classes, 
                                            classification_metrics=['accuracy', 'f1', 'precision', 'recall'],
                                            encoding_metrics=['activation_sparsity', 'binary_sparsity', 'temporal_sparsity', 'spike_density'],
                                            prog_bar_metrics=['f1'])
        
    def training_step(self, batch, batch_idx):
        x, y = batch

        preds, embeddings = self(x)

        if isinstance(self.loss_fn, CE_L1Loss):
            loss = self.loss_fn(preds, embeddings, x, y)
        elif self.loss_fn.__class__.__name__ == 'CE_SpikeSparsityLoss':
            loss = self.loss_fn(preds, embeddings, self.current_epoch, y)
        else:
            loss = self.loss_fn(preds, y)
        
        if self.backend == 'spikingjelly':
            SF.reset_net(self.model)

        self.metric_manager.track_step(loss, preds, y, embeddings, "train")

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        preds, embeddings = self(x)
        
        if isinstance(self.loss_fn, CE_L1Loss):
            loss = self.loss_fn(preds, embeddings, x, y)
            
        elif self.loss_fn.__class__.__name__ == 'CE_SpikeSparsityLoss':
            loss = self.loss_fn(preds, embeddings, self.current_epoch, y)
        else:
            loss = self.loss_fn(preds, y)
        
        if self.backend == 'spikingjelly':
            SF.reset_net(self.model)

        self.metric_manager.track_step(loss, preds, y, embeddings, "val")

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        
        preds, embeddings = self(x)
        
        if isinstance(self.loss_fn, CE_L1Loss):
            loss = self.loss_fn(preds, embeddings, x, y)
            
        elif self.loss_fn.__class__.__name__ == 'CE_SpikeSparsityLoss':
            loss = self.loss_fn(preds, embeddings, self.current_epoch, y)
        else:
            loss = self.loss_fn(preds, y)
        
        if self.backend == 'spikingjelly':
            SF.reset_net(self.model)

        self.metric_manager.track_step(loss, preds, y, embeddings, "test")

        return loss

    def on_train_epoch_end(self):
        lr_scheduler = self.lr_schedulers()
        if lr_scheduler is not None and not isinstance(lr_scheduler, list):
            current_lr = lr_scheduler.get_last_lr()[0]
            self.log('learning_rate', current_lr, on_epoch=True, prog_bar=True)

    