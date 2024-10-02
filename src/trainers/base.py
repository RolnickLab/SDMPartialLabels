import pytorch_lightning as pl
import torch
from torch import nn, optim

from src.losses import CustomCrossEntropyLoss
from src.metrics import get_metrics


class BaseTrainer(pl.LightningModule):
    def __init__(self, config):
        super(BaseTrainer, self).__init__()
        self.sigmoid_activation = nn.Sigmoid()
        self.loss_fn = CustomCrossEntropyLoss()
        self.config = config
        self.learning_rate = self.config.experiment.module.lr
        self.num_species = self.config.data.total_species
        self.class_indices_to_test = None

        # metrics to report
        metrics = get_metrics(self.config)
        for name, value, _ in metrics:
            setattr(self, "val_" + name, value)
        for name, value, _ in metrics:
            setattr(self, "train_" + name, value)
        for name, value, _ in metrics:
            setattr(self, "test_" + name, value)
        self.metrics = metrics

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            weight_decay=0.1,
        )

        return optimizer

    def __log_metric(self, mode, pred, y, mask=None):
        for name, _, scale in self.metrics:
            nname = str(mode) + "_" + name
            if name == "accuracy":
                value = getattr(self, nname)(pred, y.type(torch.uint8))
            elif name == "r2":
                value = torch.mean(getattr(self, nname)(y, pred))
            elif name == "mae" and mask != None:
                value = getattr(self, nname)(y, pred, mask=mask)
            elif name == "nonzero_mae" and mask != None:
                value = getattr(self, nname)(y, pred, mask=mask)
            elif name == "mse" and mask != None:
                value = getattr(self, nname)(y, pred, mask=mask)
            elif name == "nonzero_mse" and mask != None:
                value = getattr(self, nname)(y, pred, mask=mask)
            else:
                value = getattr(self, nname)(y, pred)

            self.log(nname, value, on_epoch=True)

    def log_metrics(self, mode, pred, y, mask=None):
        """
        log metrics through logger
        """

        unknown_mask = None
        if mask is not None:
            unknown_mask = mask.clone()
            if self.config.Ctran.use:
                unknown_mask[mask == -1] = 1
                unknown_mask[mask != -1] = 0

            loss = self.loss_fn(pred=pred, target=y, mask=unknown_mask)

            pred = pred * unknown_mask
            y = y * unknown_mask

        else:
            loss = self.loss_fn(pred, y)

        self.__log_metric(mode, pred, y, mask=unknown_mask)
        self.log(str(mode) + "_loss", loss, on_epoch=True)
