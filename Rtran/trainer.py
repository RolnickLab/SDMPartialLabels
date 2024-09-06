"""
Trainer for the Rtran framework
"""

import pickle

import pytorch_lightning as pl
from torch import nn, optim

from Rtran.dataloader import *
from Rtran.losses import (BCE, CustomCrossEntropyLoss, CustomFocalLoss,
                          RMSLELoss)
from Rtran.metrics import get_metrics
from Rtran.rtran import RTranModel


class RegressionTransformerTask(pl.LightningModule):
    def __init__(self, opts, **kwargs: Any) -> None:
        """
        opts: configurations
        """
        super().__init__()
        self.config = opts

        self.num_species = self.config.data.total_species

        # model and optimizer utils
        self.learning_rate = self.config.experiment.module.lr
        self.criterion = self.__loss_mapping(self.config.losses.criterion)

        self.input_channels = 1
        self.model = RTranModel(
            num_classes=self.num_species,
            species_list=os.path.join(
                self.config.data.files.base, self.config.data.files.species_list
            ),
            backbone=self.config.Rtran.backbone,
            pretrained_backbone=self.config.Rtran.pretrained_backbone,
            quantized_mask_bins=self.config.Rtran.quantized_mask_bins,
            input_channels=self.input_channels,
            d_hidden=self.config.Rtran.features_size,
            use_pos_encoding=self.config.Rtran.use_positional_encoding,
        )

        if self.config.experiment.module.resume:
            self.load_rtran_weights()

        self.sigmoid_activation = nn.Sigmoid()
        # self.load_resnet18_weights()
        # if using range maps (RM)
        if self.config.data.correction_factor.thresh:
            with open(
                os.path.join(
                    self.config.data.files.base,
                    self.config.data.files.correction_thresh,
                ),
                "rb",
            ) as f:
                self.RM_correction_data = pickle.load(f)

        # metrics to report
        metrics = get_metrics(self.config)
        for name, value, _ in metrics:
            setattr(self, "val_" + name, value)
        for name, value, _ in metrics:
            setattr(self, "train_" + name, value)
        for name, value, _ in metrics:
            setattr(self, "test_" + name, value)
        self.metrics = metrics

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
            self.config.Rtran.masked_loss
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

        if self.config.Rtran.mask_eval_metrics:
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
        ) or self.config.Rtran.mask_eval_metrics:
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

    def configure_optimizers(self):
        optimizer_mapping = {
            "Adam": optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.learning_rate,
                weight_decay=0.01,
            ),
            "AdamW": optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.learning_rate,
                weight_decay=0.1,
            ),  # 0.1
            "SGD": optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.learning_rate,
                momentum=0.9,
            ),
        }
        optimizer = optimizer_mapping.get(self.config.optimizer)

        if self.config.scheduler.name == "ReduceLROnPlateau":
            # scheduler_warmup = WarmupLinearSchedule(optimizer, 1, 300000)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.config.scheduler.reduce_lr_plateau.factor,
                patience=self.config.scheduler.reduce_lr_plateau.lr_schedule_patience,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "frequency": 1,
                },
            }
        else:
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

            loss = self.criterion(pred, y, mask=unknown_mask)

            pred = pred * unknown_mask
            y = y * unknown_mask

        else:
            loss = self.criterion(pred, y)

        if self.config.data.target.type == "binary":
            pred = self.sigmoid_activation(pred)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0

        self.__log_metric(mode, pred, y, mask=unknown_mask)
        self.log(str(mode) + "_loss", loss, on_epoch=True)
