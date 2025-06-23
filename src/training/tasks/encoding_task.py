import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.optim import Adam

from src.training.loss.EncoderLoss import EncoderLoss
from src.training.tasks.base_task import BaseTask
from src.training.metrics.MetricManager import MetricManager

class EncodingTask(BaseTask):

    def __init__(self, model, loss_fn, learning_rate):
        super(EncodingTask, self).__init__(model, loss_fn, learning_rate)
        self.save_hyperparameters("model", "loss_fn", "learning_rate")

        self.model = model.cuda()
        self.loss_fn = loss_fn.cuda()
        self.learning_rate = learning_rate
        
        self.metric_manager = MetricManager(log_fn = self.log, num_classes=self.num_classes, 
                                            classification_metrics=[],
                                            encoding_metrics=['activation_sparsity', 'binary_sparsity', 'temporal_sparsity', 'spike_density'],
                                            prog_bar_metrics=['activation_sparsity'])

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        
        if isinstance(self.loss_fn, EncoderLoss):
            x = x.flatten(start_dim=1)
            
        loss = self.loss_fn(preds, x)

        self.metric_manager.track_step(loss, preds, y, preds[0], "train")

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        
        if isinstance(self.loss_fn, EncoderLoss):
            x = x.flatten(start_dim=1)
            
        loss = self.loss_fn(preds, x)

        self.metric_manager.track_step(loss, preds, y, preds[0], "val")

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        
        if isinstance(self.loss_fn, EncoderLoss):
            x = x.flatten(start_dim=1)
            
        loss = self.loss_fn(preds, x)
        
        self.metric_manager.track_step(loss, preds, y, preds[0], "test")

        return loss
