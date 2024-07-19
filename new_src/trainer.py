import pytorch_lightning as pl
import torch
from new_src.utils import multi_label_accuracy
from new_src.models import *


class sPlotsTrainer(pl.LightningModule):
    def __init__(self, config):
        super(sPlotsTrainer, self).__init__()
        self.learning_rate = config.training.learning_rate
        self.activation = nn.Sigmoid()
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Create model
        self.model = globals()[config.model.name](
            input_dim=config.model.input_dim,
            hidden_dim=config.model.hidden_dim,
            output_dim=config.model.output_dim,
        )

    def training_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.activation(self.model(data))
        loss = self.loss_fn(outputs, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.model(data)

        loss = self.loss_fn(outputs, targets)
        accuracy = multi_label_accuracy(outputs, targets)
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)

    def testing_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.activation(self.model(data))

        loss = self.loss_fn(outputs, targets)
        accuracy = multi_label_accuracy(outputs, targets)
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


"""
- input normalization
"""