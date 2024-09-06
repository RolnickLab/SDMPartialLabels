import os

import numpy as np

from Rtran.metrics import get_metrics
from Rtran.models import SimpleMLP
from src.trainers.base import BaseTrainer
from src.utils import satbird_species_split


class MLPTrainer(BaseTrainer):
    def __init__(self, config):
        super(MLPTrainer, self).__init__(config)

        self.model = SimpleMLP(
            input_dim=self.config.experiment.module.input_dim,
            hidden_dim=self.config.experiment.module.hidden_dim,
            output_dim=self.config.data.total_species,
        )
        self.indices_to_predict = None
        out_num_classes = self.config.data.total_species

        if self.config.predict_family_of_species != -1 and self.config.data.species is None:
            self.indices_to_predict = satbird_species_split(index=self.config.predict_family_of_species,
                                                            base_data_folder=self.config.data.files.base)

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
        data = batch["data"]
        targets = batch["target"]

        predictions = self.sigmoid_activation(self.model(data))

        loss = self.loss_fn(pred=predictions, target=targets)

        self.log_metrics(mode="train", pred=predictions, y=targets)

        return loss

    def validation_step(self, batch, batch_idx):
        data = batch["data"]
        targets = batch["target"]

        predictions = self.sigmoid_activation(self.model(data))

        if self.indices_to_predict is not None:
            predictions = predictions[:, self.indices_to_predict]
            targets = targets[:, self.indices_to_predict]

        self.log_metrics(mode="val", pred=predictions, y=targets)

    def test_step(self, batch, batch_idx):
        data = batch["data"]
        targets = batch["target"]

        predictions = self.sigmoid_activation(self.model(data))

        if self.indices_to_predict is not None:
            predictions = predictions[:, self.indices_to_predict]
            targets = targets[:, self.indices_to_predict]

        self.log_metrics(mode="test", pred=predictions, y=targets)
