import os

import torch
from torchmetrics.classification import MultilabelAUROC

from src.models.baselines import SimpleMLP
from src.trainers.base import BaseTrainer
from src.utils import eval_species_split


class BaselineTrainer(BaseTrainer):
    def __init__(self, config):
        super(BaselineTrainer, self).__init__(config)

        self.model = SimpleMLP(
            input_dim=self.config.model.input_dim,
            hidden_dim=self.config.model.hidden_dim,
            num_classes=self.num_species,
        )

        if self.config.predict_family_of_species != -1:
            self.class_indices_to_test = eval_species_split(
                index=self.config.predict_family_of_species,
                base_data_folder=os.path.join(
                    self.config.data.files.base,
                    self.config.data.files.satbird_species_indices_path,
                ),
                multi_taxa=self.config.data.multi_taxa,
                per_taxa_species_count=self.config.data.per_taxa_species_count,
            )

        if self.class_indices_to_test is not None:
            self.num_species = len(self.class_indices_to_test)

        print(f"Number of classes: {self.num_species}")
        if self.config.data.multi_taxa and "plant" in self.config.data.per_taxa_species_count.keys():
            self.test_auc_metric = MultilabelAUROC(
                num_labels=self.num_species, average="macro"
            )

    def training_step(self, batch, batch_idx):
        data = batch["data"]
        targets = batch["targets"]
        available_species_mask = batch["available_species_mask"]

        if not torch.eq(available_species_mask, 0).any():
            available_species_mask = None

        predictions = self.model(data)

        loss = self.loss_fn(
            pred=predictions, target=targets, mask=available_species_mask
        )
        predictions = self.sigmoid_activation(self.model(data))

        self.log_metrics(
            mode="train", pred=predictions, y=targets, mask=available_species_mask
        )

        return loss

    def validation_step(self, batch, batch_idx):
        data = batch["data"]
        targets = batch["targets"]
        available_species_mask = batch["available_species_mask"]

        if not torch.eq(available_species_mask, 0).any():
            available_species_mask = None

        predictions = self.sigmoid_activation(self.model(data))

        self.log_metrics(
            mode="val", pred=predictions, y=targets, mask=available_species_mask
        )

    def test_step(self, batch, batch_idx):
        data = batch["data"]
        targets = batch["targets"]
        available_species_mask = batch["available_species_mask"]

        if not torch.eq(available_species_mask, 0).any():
            available_species_mask = None

        predictions = self.sigmoid_activation(self.model(data))

        if self.class_indices_to_test is not None:
            predictions = predictions[:, self.class_indices_to_test]
            targets = targets[:, self.class_indices_to_test]

        if self.config.data.multi_taxa and "plant" in self.config.data.per_taxa_species_count.keys() and self.config.predict_family_of_species == 1:
            self.test_auc_metric.update(predictions, targets.long())
        else:
            self.log_metrics(
                mode="test", pred=predictions, y=targets, mask=available_species_mask
            )

    def on_test_epoch_end(self):
        if self.config.data.multi_taxa and "plant" in self.config.data.per_taxa_species_count.keys() and self.config.predict_family_of_species == 1:
            self.log("test_auroc", self.test_auc_metric.compute())
            self.test_auc_metric.reset()