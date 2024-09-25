"""
Trainer for the ctran framework
"""

from torch import nn

from src.dataloaders.dataloader import *
from src.losses import BCE, CustomCrossEntropyLoss, CustomFocalLoss, RMSLELoss
from src.models.ctran import CTranModel
from src.trainers.base import BaseTrainer


class CTranTrainer(BaseTrainer):
    def __init__(self, config, **kwargs: Any) -> None:
        """
        opts: configurations
        """
        super(CTranTrainer, self).__init__(config)

        self.num_species = self.config.data.total_species

        # model and optimizer utils
        self.criterion = self.__loss_mapping(self.config.losses.criterion)

        self.input_channels = 1
        self.model = CTranModel(
            num_classes=self.num_species,
            species_list=os.path.join(
                self.config.data.files.base, self.config.data.files.species_list
            ),
            backbone=self.config.Ctran.backbone,
            pretrained_backbone=self.config.Ctran.pretrained_backbone,
            quantized_mask_bins=self.config.Ctran.quantized_mask_bins,
            input_channels=self.input_channels,
            d_hidden=self.config.Ctran.features_size,
            use_pos_encoding=self.config.Ctran.use_positional_encoding,
        )

        if self.config.experiment.module.resume:
            self.load_rtran_weights()

    def load_resnet18_weights(self):
        ckpt = torch.load(self.config.experiment.module.resume)
        loaded_dict = ckpt["state_dict"]
        model_dict = self.model.state_dict()

        # load state dict keys
        for key_model, key_pretrained in zip(model_dict.keys(), loaded_dict.keys()):
            # ignore first layer weights(use imagenet ones)
            if "backbone" in key_model:
                if "fc" in key_model:
                    continue
                print("loaded.. ", key_model, key_pretrained)
                model_dict[key_model] = loaded_dict[key_pretrained]

        self.model.load_state_dict(model_dict)

    def load_rtran_weights(self):
        ckpt = torch.load(self.config.experiment.module.resume)
        loaded_dict = ckpt["state_dict"]
        model_dict = self.model.state_dict()

        # load state dict keys
        for key_model, key_pretrained in zip(model_dict.keys(), loaded_dict.keys()):
            # ignore first layer weights(use imagenet ones)
            if "output_linear" in key_model or "label_embeddings" in key_model:
                continue
            print("loaded.. ", key_model, key_pretrained)
            model_dict[key_model] = loaded_dict[key_pretrained]

        self.model.load_state_dict(model_dict)

    def training_step(self, batch: Dict[str, Any], batch_idx: int):

        hotspot_id = batch["hotspot_id"]

        x = batch["data"]
        y = batch["targets"]
        mask = batch["mask"].long()

        if self.config.data.target.type == "binary":
            y_pred = self.model(x, mask.clone(), batch["mask_q"])
        else:
            y_pred = self.sigmoid_activation(
                self.model(x, mask.clone(), batch["mask_q"])
            )

        # if using range maps for SatBird
        if self.config.data.correction_factor.thresh:
            range_maps_correction_data = (
                self.RM_correction_data.reset_index()
                .set_index("hotspot_id")
                .drop(columns=["index"])
            )
            range_maps_correction_data = range_maps_correction_data.reindex(
                list(hotspot_id), fill_value=True
            ).values
            range_maps_correction_data = torch.tensor(
                range_maps_correction_data, device=y.device
            )
            ones = torch.ones(
                range_maps_correction_data.shape[0],
                self.config.data.species[1],
                device=y.device,
            )
            range_maps_correction_data = torch.cat(
                (range_maps_correction_data, ones), 1
            )
            # y_pred[:, :RM_end_index] = y_pred[:, :RM_end_index] * range_maps_correction_data.int()
            # y[:, :RM_end_index] = y[:, :RM_end_index] * range_maps_correction_data.int()
            y_pred = y_pred * range_maps_correction_data.int()
            y = y * range_maps_correction_data.int()

        print(mask.unique())
        if (
            self.config.Ctran.masked_loss
        ):  # to consider unknown labels only for the loss
            unknown_mask = mask.clone()
            unknown_mask[mask != -1] = 0
            unknown_mask[mask == -1] = 1
        elif -2 in mask:  # to mask out species with no targets from the loss
            unknown_mask = mask.clone()
            unknown_mask[mask == -2] = 0
            unknown_mask[mask == -1] = 1
        else:
            unknown_mask = None

        loss = self.criterion(y_pred, y, mask=unknown_mask)
        if batch_idx % 50 == 0:
            self.log("train_loss", loss, on_epoch=True)
            if self.config.data.species is not None:
                if len(self.config.data.species) > 1:
                    self.log_metrics(mode="train", pred=y_pred, y=y, mask=mask)
            else:
                self.log_metrics(mode="train", pred=y_pred, y=y)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:

        hotspot_id = batch["hotspot_id"]

        x = batch["data"]
        y = batch["targets"]
        mask = batch["mask"].long()

        if self.config.data.target.type == "binary":
            y_pred = self.model(x, mask.clone(), batch["mask_q"])
        else:
            y_pred = self.sigmoid_activation(
                self.model(x, mask.clone(), batch["mask_q"])
            )

        # if using range maps for SatBird
        if self.config.data.correction_factor.thresh:
            range_maps_correction_data = (
                self.RM_correction_data.reset_index()
                .set_index("hotspot_id")
                .drop(columns=["index"])
            )
            range_maps_correction_data = range_maps_correction_data.reindex(
                list(hotspot_id), fill_value=True
            ).values
            range_maps_correction_data = torch.tensor(
                range_maps_correction_data, device=y.device
            )
            ones = torch.ones(
                range_maps_correction_data.shape[0],
                self.config.data.species[1],
                device=y.device,
            )
            range_maps_correction_data = torch.cat(
                (range_maps_correction_data, ones), 1
            )
            y_pred = y_pred * range_maps_correction_data.int()
            y = y * range_maps_correction_data.int()

        if self.config.Ctran.mask_eval_metrics:
            self.log_metrics(mode="val", pred=y_pred, y=y, mask=mask)
        elif self.config.data.species is not None:
            if len(self.config.data.species) > 1:
                self.log_metrics(mode="val", pred=y_pred, y=y, mask=mask)
            else:
                self.log_metrics(mode="val", pred=y_pred, y=y)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Test step"""
        hotspot_id = batch["hotspot_id"]

        x = batch["data"]
        y = batch["targets"]
        mask = batch["mask"].long()

        eval_mask = batch["eval_mask"].long()

        if self.config.data.target.type == "binary":
            y_pred = self.model(x, mask.clone(), batch["mask_q"])
        else:
            y_pred = self.sigmoid_activation(
                self.model(x, mask.clone(), batch["mask_q"])
            )

        # if using range maps for SatBird
        if self.config.data.correction_factor.thresh:
            range_maps_correction_data = (
                self.RM_correction_data.reset_index()
                .set_index("hotspot_id")
                .drop(columns=["index"])
            )
            range_maps_correction_data = range_maps_correction_data.reindex(
                list(hotspot_id), fill_value=True
            ).values
            range_maps_correction_data = torch.tensor(
                range_maps_correction_data, device=y.device
            )
            ones = torch.ones(
                range_maps_correction_data.shape[0],
                self.config.data.species[1],
                device=y.device,
            )
            range_maps_correction_data = torch.cat(
                (range_maps_correction_data, ones), 1
            )
            y_pred = y_pred * range_maps_correction_data.int()
            y = y * range_maps_correction_data.int()

        if (
            self.config.data.species is not None and len(self.config.data.species) > 1
        ) or self.config.Ctran.mask_eval_metrics:
            self.log_metrics(mode="test", pred=y_pred, y=y, mask=eval_mask)

        else:
            self.log_metrics(mode="test", pred=y_pred, y=y)

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
