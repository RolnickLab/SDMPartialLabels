import os

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim

from Rtran.losses import CustomCrossEntropyLoss
from Rtran.metrics import get_metrics
from Rtran.models import SimpleMLP


class BaseTrainer(pl.LightningModule):
    def __init__(self, config):
        super(BaseTrainer, self).__init__()
        self.sigmoid_activation = nn.Sigmoid()
        self.loss_fn = CustomCrossEntropyLoss()
        self.config = config
        self.learning_rate = self.config.experiment.module.lr

        self.model = SimpleMLP(
            input_dim=self.config.experiment.module.input_dim,
            hidden_dim=self.config.experiment.module.hidden_dim,
            output_dim=self.config.data.total_species,
        )
        self.indices_to_predict = None
        out_num_classes = self.config.data.total_species

        if self.config.predict_family_of_species != -1:
            songbird_indices = [
                "stats/nonsongbird_indices.npy",
                "stats/songbird_indices.npy",
            ]
            self.indices_to_predict = np.load(os.path.join(self.config.data.files.base, songbird_indices[self.config.predict_family_of_species]))

        if self.indices_to_predict is not None:
            out_num_classes = len(self.indices_to_predict)

        print(f"Number of classes: {out_num_classes}")

        # metrics to report
        metrics = get_metrics(self.config)
        for name, value, _ in metrics:
            setattr(self, "val_" + name, value)
        for name, value, _ in metrics:
            setattr(self, "train_" + name, value)
        for name, value, _ in metrics:
            setattr(self, "test_" + name, value)
        self.metrics = metrics

    def training_step(self, batch, batch_idx):
        data = batch["env"]
        targets = batch["target"]

        predictions = self.sigmoid_activation(self.model(data))

        loss = self.loss_fn(pred=predictions, target=targets)

        self.log_metrics(mode="train", pred=predictions, y=targets)

        return loss

    def validation_step(self, batch, batch_idx):
        data = batch["env"]
        targets = batch["target"]

        predictions = self.sigmoid_activation(self.model(data))

        if self.indices_to_predict is not None:
            predictions = predictions[:, self.indices_to_predict]
            targets = targets[:, self.indices_to_predict]

        self.log_metrics(mode="val", pred=predictions, y=targets)

    def test_step(self, batch, batch_idx):
        data = batch["env"]
        targets = batch["target"]

        predictions = self.sigmoid_activation(self.model(data))

        if self.indices_to_predict is not None:
            predictions = predictions[:, self.indices_to_predict]
            targets = targets[:, self.indices_to_predict]

        self.log_metrics(mode="test", pred=predictions, y=targets)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                weight_decay=0.1)

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
            if self.config.Rtran.mask_eval_metrics:
                unknown_mask[mask == -1] = 1
                unknown_mask[mask != -1] = 0
            else:
                unknown_mask[mask == -1] = 1
                unknown_mask[mask == -2] = 0

            loss = self.loss_fn(pred=pred, target=y, mask=unknown_mask)

            pred = pred * unknown_mask
            y = y * unknown_mask

        else:
            loss = self.loss_fn(pred, y)

        self.__log_metric(mode, pred, y, mask=unknown_mask)
        self.log(str(mode) + "_loss", loss, on_epoch=True)