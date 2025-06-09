"""
Trainer for the ctran framework
"""

import inspect

from src.dataloaders.dataloader import *
from src.trainers.base import BaseTrainer
from src.utils import eval_species_split


class SDMPartialTrainer(BaseTrainer):
    def __init__(self, config, **kwargs: Any) -> None:
        """
        opts: configurations
        """
        super(SDMPartialTrainer, self).__init__(config)

        model_kwargs = {
            "input_dim": self.config.model.input_dim,
            "hidden_dim": self.config.model.hidden_dim,
            "num_classes": self.num_species,
            "quantized_mask_bins": self.config.partial_labels.quantized_mask_bins,
            "backbone": self.config.model.backbone,
            "n_attention_layers": self.config.model.n_attention_layers,
            "n_heads": self.config.model.n_heads,
            "dropout": self.config.model.dropout,
            "n_backbone_layers": self.config.model.n_backbone_layers,
            "tokenize_state": self.config.partial_labels.tokenize_state,
            "use_unknown_token": self.config.partial_labels.use_unknown_token,
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

        # if eval_known_rate == 0, everything is unknown, but we want to predict certain families
        if (
            self.config.predict_family_of_species != -1
        ):
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

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        x = batch["data"]
        y = batch["targets"]
        mask = batch["mask"].long()

        y_pred = self.model(x, batch["mask_q"])

        # loss function does sigmoid + cross entropy
        loss = self.loss_fn(y_pred, y, mask=batch["available_species_mask"].long())
        y_pred = self.sigmoid_activation(y_pred)

        self.log("train_loss", loss, on_epoch=True)

        if batch_idx % 50 == 0:
            self.log_metrics(mode="train", pred=y_pred, y=y, mask=mask)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        x = batch["data"]
        y = batch["targets"]
        mask = batch["mask"].long()

        y_pred = self.sigmoid_activation(self.model(x, batch["mask_q"]))

        self.log_metrics(mode="val", pred=y_pred, y=y, mask=mask, multi_taxa=self.config.data.multi_taxa,
                         per_taxa_species_count=self.config.data.per_taxa_species_count)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Test step"""
        x = batch["data"]
        y = batch["targets"]
        mask = batch["mask"].long()

        y_pred = self.sigmoid_activation(self.model(x, batch["mask_q"]))

        if self.class_indices_to_test is not None:
            y_pred = y_pred[:, self.class_indices_to_test]
            y = y[:, self.class_indices_to_test]
            mask = mask[:, self.class_indices_to_test]

        if self.config.data.multi_taxa and "plant" in self.config.data.per_taxa_species_count.keys() and self.config.predict_family_of_species == 1:
            self.test_auc_metric.update(y_pred[:, self.plant_test_species_indices], y[:, self.plant_test_species_indices].long())
        else:
            self.log_metrics(mode="test", pred=y_pred, y=y, mask=mask)

        # saving model predictions
        if self.config.save_preds_path != "":
            for i, elem in enumerate(y_pred):
                np.save(
                    os.path.join(
                        self.config.base_dir,
                        self.config.save_preds_path,
                        batch["hotspot_id"][i] + ".npy",
                    ),
                    elem.cpu().detach().numpy(),
                )

    def on_test_epoch_end(self):
        if self.config.data.multi_taxa and "plant" in self.config.data.per_taxa_species_count.keys() and self.config.predict_family_of_species == 1:
            self.log("test_auroc", self.test_auc_metric.compute())
            self.test_auc_metric.reset()
