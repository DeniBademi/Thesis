from collections import defaultdict
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, F1Score, Precision, Recall
from src.training.metrics.sparsity_metrics import SpikeSparsity
import torch




class MetricManager(LightningModule):
    
    supported_classification_metrics = {
        "accuracy": lambda x: Accuracy(task='multiclass', num_classes=x).cuda(),
        "f1": lambda x: F1Score(task='multiclass', num_classes=x).cuda(),
        "precision": lambda x: Precision(task='multiclass', num_classes=x).cuda(),
        "recall": lambda x: Recall(task='multiclass', num_classes=x).cuda(),
    }
    
    supported_encoding_metrics = {
        "activation_sparsity": SpikeSparsity.activation_sparsity,
        "binary_sparsity": SpikeSparsity.binary_sparsity,
        "temporal_sparsity": SpikeSparsity.temporal_sparsity,
        "spike_density": SpikeSparsity.spike_density
    }
    
    def __init__(self, log_fn, num_classes, classification_metrics, encoding_metrics, prog_bar_metrics=[]):
        super(MetricManager, self).__init__()
        self.log = log_fn
        self.metrics_to_track = classification_metrics + encoding_metrics
        self.classification_metrics = {}
        self.encoding_metrics = {}
        assert all(metric in self.metrics_to_track for metric in prog_bar_metrics), "All prog_bar_metrics must be in metrics_to_track"
        
        self.prog_bar_metrics = prog_bar_metrics
        self.step_outputs = defaultdict(list)
        
        for metric in self.metrics_to_track:
            if metric in self.supported_classification_metrics:
                self.classification_metrics[metric] = self.supported_classification_metrics[metric](num_classes)
            elif metric in self.supported_encoding_metrics:
                self.encoding_metrics[metric] = self.supported_encoding_metrics[metric]
            else:
                raise ValueError(f"Metric {metric} is not supported")
        
        self.running_loss_train = 0.0
        self.running_loss_val = 0.0

    def track_step(self, loss, preds, y, embeddings, mode):
        assert mode in ["train", "val", "test"]
        
        curr_metrics = {}
        self.log(f"{mode}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        for key, metric in self.classification_metrics.items():
            curr_metrics[key] = metric(preds, y)
            self.log(f"{mode}_{key}", curr_metrics[key], 
                     on_step=True,
                     on_epoch='loss' in key or 'accuracy' in key,
                     prog_bar=key in self.prog_bar_metrics)
        
        for key, metric in self.encoding_metrics.items():
            curr_metrics[key] = metric(embeddings)
            self.log(f"{mode}_{key}", curr_metrics[key], 
                     on_step=True,
                     on_epoch='sparsity' in key or 'density' in key,
                     prog_bar=key in self.prog_bar_metrics)

        # self.step_outputs['loss'].append(loss)
        # for key, value in curr_metrics.items():
        #     self.step_outputs[key].append(value)
        
        return curr_metrics
        
    # def end_epoch(self, mode):
        
    #     return mean_step_outputs
