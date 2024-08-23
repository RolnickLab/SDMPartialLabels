import pytorch_lightning as pl
import torch
from torchmetrics.classification import MultilabelAUROC

from new_src.maskedbaseline import *
from new_src.models import *
from new_src.utils import multi_label_accuracy, trees_masking
from Rtran.rtran import RTranModel


class sPlotsTrainer(pl.LightningModule):
    def __init__(self, config):
        super(sPlotsTrainer, self).__init__()
        self.learning_rate = config.training.learning_rate
        self.activation = nn.Sigmoid()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.config = config

        # Create model
        if config.data.partial_labels:
            self.model = globals()[self.config.model.name](
                num_classes=self.config.model.output_dim,
                backbone=self.config.model.backbone,
                input_channels=self.config.model.input_dim,
                d_hidden=self.config.model.hidden_dim,
            )
        else:
            self.model = globals()[self.config.model.name](
                input_channels=self.config.model.input_dim,
                hidden_dim=self.config.model.hidden_dim,
                output_dim=self.config.model.output_dim,
            )

        self.indices_to_predict = trees_masking(config=self.config.data)

        out_num_classes = self.config.model.output_dim
        if self.indices_to_predict is not None:
            out_num_classes = len(self.indices_to_predict)

        print(f"Number of classes: {out_num_classes}")
        self.val_auc_metric = MultilabelAUROC(
            num_labels=out_num_classes, average="macro"
        )
        self.test_auc_metric = MultilabelAUROC(
            num_labels=out_num_classes, average="macro"
        )

    def training_step(self, batch, batch_idx):
        data, targets, mask = batch
        if mask is not None:
            predictions = self.model(data, mask)
        else:
            predictions = self.model(data)

        loss = self.loss_fn(predictions, targets)
        self.logger.log_metrics({"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        data, targets, mask = batch

        if mask is not None:
            predictions = self.model(data, mask)
        else:
            predictions = self.model(data)

        if self.indices_to_predict is not None:
            predictions = predictions[:, self.indices_to_predict]
            targets = targets[:, self.indices_to_predict]

        loss = self.loss_fn(predictions, targets)

        predictions = self.activation(predictions)

        accuracy = multi_label_accuracy(predictions, targets)

        self.logger.log_metrics({"val_loss": loss})
        self.logger.log_metrics({"val_accuracy": accuracy})

        self.val_auc_metric.update(predictions, targets.long())
        self.log("val_auroc", self.val_auc_metric, on_epoch=True)

    def test_step(self, batch, batch_idx):
        data, targets, mask = batch

        if mask is not None:
            predictions = self.model(data, mask)
        else:
            predictions = self.model(data)

        if self.indices_to_predict is not None:
            predictions = predictions[:, self.indices_to_predict]
            targets = targets[:, self.indices_to_predict]

        loss = self.loss_fn(predictions, targets)
        predictions = self.activation(predictions)

        accuracy = multi_label_accuracy(predictions, targets)

        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)

        self.test_auc_metric.update(predictions, targets.long())
        self.log("test_auroc", self.test_auc_metric, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
