# supports training baseline models and C-tran on splot data
import inspect
import os

import numpy as np
import pytorch_lightning as pl
from torchmetrics.classification import MultilabelAUROC

from src.models.baselines import *
from src.utils import multi_label_accuracy, trees_masking


class sPlotTrainer(pl.LightningModule):
    def __init__(self, config):
        super(sPlotTrainer, self).__init__()
        self.learning_rate = config.training.learning_rate
        self.config = config
        self.all_test_predictions = []
        self.all_test_targets = []

        model_kwargs = {
            "input_dim": self.config.model.input_dim,
            "hidden_dim": self.config.model.hidden_dim,
            "num_classes": self.config.model.num_classes,
            "backbone": self.config.model.backbone,
            "quantized_mask_bins": self.config.data.partial_labels.quantized_mask_bins,
        }
        model_class = globals()[self.config.model.name]
        model_signature = inspect.signature(model_class.__init__)
        # check which args are needed for the model params
        valid_args = {
            param: value
            for param, value in model_kwargs.items()
            if param in model_signature.parameters
        }
        # Initialize the model using only the relevant arguments
        self.model = model_class(**valid_args)

        self.indices_to_predict = trees_masking(config=self.config.data)

        out_num_classes = self.config.model.num_classes
        if self.indices_to_predict is not None:
            out_num_classes = len(self.indices_to_predict)

        print(f"Number of classes: {out_num_classes}")
        self.val_auc_metric = MultilabelAUROC(
            num_labels=out_num_classes, average="macro"
        )

        self.activation = nn.Sigmoid()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, batch):
        if self.config.data.partial_labels.use:
            return self.model(batch["data"], batch["mask"])
        return self.model(batch["data"])

    def _shared_step(self, batch, stage: str):
        data, targets = batch["data"], batch["targets"]
        predictions = self(batch)

        if self.indices_to_predict is not None:
            predictions = predictions[:, self.indices_to_predict]
            targets = targets[:, self.indices_to_predict]

        loss = self.loss_fn(predictions, targets)
        predictions = self.activation(predictions)
        accuracy = multi_label_accuracy(predictions, targets)

        metrics = {
            f"{stage}_loss": loss,
            f"{stage}_accuracy": accuracy,
        }

        if stage == "val":
            self.val_auc_metric.update(predictions, targets.long())
        elif stage == "test":
            self.all_test_predictions.append(predictions)
            self.all_test_targets.append(targets.long())
            # saving model predictions
            if self.config.logger.save_preds_path != "":
                for i, elem in enumerate(predictions):
                    np.save(
                        os.path.join(
                            self.config.logger.checkpoint_path,
                            self.config.logger.experiment_name,
                            self.config.logger.save_preds_path,
                            batch["hotspot_id"][i] + ".npy",
                        ),
                        elem.cpu().detach().numpy(),
                    )

        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def on_validation_epoch_end(self):
        self.log("val_auroc", self.val_auc_metric.compute())
        self.val_auc_metric.reset()

    def on_test_epoch_end(self):
        self.all_test_predictions = torch.cat(self.all_test_predictions, dim=0)
        self.all_test_targets = torch.cat(self.all_test_targets, dim=0)
        non_zero_indices = self.all_test_targets.sum(0) != 0
        num_classes = int(non_zero_indices.long().sum(0))

        print("Present number of classes: ", num_classes)
        self.test_auc_metric = MultilabelAUROC(
            num_labels=num_classes, average="macro"
        )
        self.test_auc_metric.update(self.all_test_predictions[:, non_zero_indices], self.all_test_targets[:, non_zero_indices])
        self.log("test_auroc", self.test_auc_metric.compute())
        self.test_auc_metric.reset()
        self.all_test_predictions = []
        self.all_test_targets = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
