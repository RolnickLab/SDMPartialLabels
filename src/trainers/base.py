import os

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torchmetrics.classification import MultilabelAUROC

from src.losses import CustomCrossEntropyLoss, RMSLELoss, CustomFocalLoss, BCE
from src.metrics import get_metrics


class BaseTrainer(pl.LightningModule):
    def __init__(self, config):
        super(BaseTrainer, self).__init__()
        self.sigmoid_activation = nn.Sigmoid()
        self.config = config
        self.learning_rate = self.config.training.lr
        self.num_species = self.config.data.total_species
        self.class_indices_to_test = None

        if self.config.data.multi_taxa and "plant" in self.config.data.per_taxa_species_count.keys():
            self.plant_test_species_indices = list(np.load(os.path.join(self.config.data.files.base, self.config.data.files.plant_test_species_indices_file)))
            self.test_auc_metric = MultilabelAUROC(
                num_labels=len(self.plant_test_species_indices), average="macro"
            )

        # metrics to report
        metrics = get_metrics(self.config)
        for name, value, _ in metrics:
            setattr(self, "val_" + name, value)
        for name, value, _ in metrics:
            setattr(self, "train_" + name, value)
        for name, value, _ in metrics:
            setattr(self, "test_" + name, value)
        self.metrics = metrics

        self.loss_fn = self.__loss_mapping(self.config.losses.criterion)

    def __loss_mapping(self, loss_fn_name):
        loss_mapping = {
            "MSE": nn.MSELoss(),
            "MAE": nn.L1Loss(),
            "RMSLE": RMSLELoss(),
            "Focal": CustomFocalLoss(),
            "CE": CustomCrossEntropyLoss(),
            "BCE": BCE()
        }
        return loss_mapping.get(loss_fn_name)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            weight_decay=0.2,
        )

        return optimizer

    def __log_metric(self, mode, pred, y, mask=None, postfix=None):
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

            if postfix:
                nname = nname + "_" + postfix
            self.log(nname, value, on_epoch=True)

    def log_metrics(self, mode, pred, y, mask=None, multi_taxa=None, per_taxa_species_count=None):
        """
        log metrics through logger
        """

        unknown_mask = None
        if mask is not None:
            unknown_mask = mask.clone()
            if self.config.partial_labels.use:
                unknown_mask[mask == -1] = 1
                unknown_mask[mask != -1] = 0

            loss = self.loss_fn(pred=pred, target=y, mask=unknown_mask)

            pred = pred * unknown_mask
            y = y * unknown_mask

        else:
            loss = self.loss_fn(pred, y)
        if mode == "val" and multi_taxa:
            taxa_indices = {
                0: np.arange(0, list(per_taxa_species_count.values())[0]),  # birds
                1: np.arange(list(per_taxa_species_count.values())[0],
                             list(per_taxa_species_count.values())[0] + list(per_taxa_species_count.values())[1])
            }
            for i, k in enumerate(per_taxa_species_count.keys()):
                pred_ = pred[:, taxa_indices[i]]
                y_ = y[:, taxa_indices[i]]
                unknown_mask_ = unknown_mask[:, taxa_indices[i]]
                self.__log_metric(mode, pred=pred_, y=y_, mask=unknown_mask_, postfix=str(k))
        else:
            self.__log_metric(mode, pred, y, mask=unknown_mask)
        self.log(str(mode) + "_loss", loss, on_epoch=True)
