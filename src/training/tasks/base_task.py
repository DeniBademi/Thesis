import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

class BaseTask(LightningModule):
    """
    This class is used to wrap pytorch neural networks and provide a common interface for training and
    evaluation.
    """

    def __init__(self, model, loss_fn, learning_rate):
        super(BaseTask, self).__init__()

        self.model = model.cuda()
        self.loss_fn = loss_fn.cuda()
        self.learning_rate = learning_rate
        self.num_classes = self.model.num_classes if hasattr(self.model, "num_classes") else None

        
    def forward(self, x):
        pred = self.model(x)
        return pred

    def smoke_test(self, x):
        self.model.eval()
        with torch.no_grad():
            y_hat = self.model(x)
        return y_hat
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.2)
        return [optimizer], [lr_scheduler]

    def configure_weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
