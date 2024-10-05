"""
Trainer for the ctran framework
"""

from torch import nn

from src.dataloaders.dataloader import *
from src.losses import BCE, CustomCrossEntropyLoss, CustomFocalLoss, RMSLELoss
from src.models.ctran import CTranModel
from src.models.baselines import SimpleMLPMasked
from src.trainers.base import BaseTrainer
from src.utils import eval_species_split


class CTranTrainer(BaseTrainer):
    def __init__(self, config, **kwargs: Any) -> None:
        """
        opts: configurations
        """
        super(CTranTrainer, self).__init__(config)

        self.criterion = self.__loss_mapping(self.config.losses.criterion)

        self.model = globals()[self.config.model.name](
            input_channels=self.config.model.input_dim,
            d_hidden=self.config.model.hidden_dim,
            num_classes=self.num_species,
            backbone=self.config.model.backbone,
            quantized_mask_bins=self.config.partial_labels.quantized_mask_bins,
        )

        # if eval_known_rate == 0, everything is unknown, but we want to predict certain families
        if (
            self.config.partial_labels.eval_known_ratio == 0
            and self.config.predict_family_of_species != -1
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

        y_pred = self.sigmoid_activation(self.model(x, mask.clone(), batch["mask_q"]))

        loss = self.criterion(y_pred, y, mask=batch["available_species_mask"].long())
        self.log("train_loss", loss, on_epoch=True)

        if batch_idx % 50 == 0:
            self.log_metrics(mode="train", pred=y_pred, y=y, mask=mask)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        x = batch["data"]
        y = batch["targets"]
        mask = batch["mask"].long()

        y_pred = self.sigmoid_activation(self.model(x, mask.clone(), batch["mask_q"]))

        self.log_metrics(mode="val", pred=y_pred, y=y, mask=mask)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Test step"""
        x = batch["data"]
        y = batch["targets"]
        mask = batch["mask"].long()

        y_pred = self.sigmoid_activation(self.model(x, mask.clone(), batch["mask_q"]))

        if self.class_indices_to_test is not None:
            y_pred = y_pred[:, self.class_indices_to_test]
            y = y[:, self.class_indices_to_test]
            mask = mask[:, self.class_indices_to_test]

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

    def __loss_mapping(self, loss_fn_name):
        loss_mapping = {
            "MSE": nn.MSELoss(),
            "MAE": nn.L1Loss(),
            "RMSLE": RMSLELoss(),
            "Focal": CustomFocalLoss(),
            "CE": CustomCrossEntropyLoss(),
            "BCE": BCE(),
        }
        return loss_mapping.get(loss_fn_name)
