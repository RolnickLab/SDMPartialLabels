from src.metrics import get_metrics
from src.models.baselines import SimpleMLP
from src.trainers.base import BaseTrainer
from src.utils import eval_species_split


class MLPTrainer(BaseTrainer):
    def __init__(self, config):
        super(MLPTrainer, self).__init__(config)

        self.model = SimpleMLP(
            input_dim=self.config.experiment.module.input_dim,
            hidden_dim=self.config.experiment.module.hidden_dim,
            output_dim=self.config.data.total_species,
        )
        self.class_indices_to_eval = None
        out_num_classes = self.config.data.total_species

        if self.config.predict_family_of_species != -1:
            self.class_indices_to_eval = eval_species_split(
                index=self.config.predict_family_of_species,
                base_data_folder=self.config.data.files.base,
                species_set=self.species_set,
            )

        if self.class_indices_to_eval is not None:
            out_num_classes = len(self.class_indices_to_eval)

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
        targets = batch["targets"]
        species_mask = batch["species_mask"]

        predictions = self.sigmoid_activation(self.model(data))

        loss = self.loss_fn(pred=predictions, target=targets, mask=species_mask)
        self.log_metrics(mode="train", pred=predictions, y=targets, mask=species_mask)

        return loss

    def validation_step(self, batch, batch_idx):
        data = batch["data"]
        targets = batch["targets"]
        species_mask = batch["species_mask"]

        predictions = self.sigmoid_activation(self.model(data))

        if self.class_indices_to_eval is not None:
            predictions = predictions[:, self.class_indices_to_eval]
            targets = targets[:, self.class_indices_to_eval]

        self.log_metrics(mode="val", pred=predictions, y=targets, mask=species_mask)

    def test_step(self, batch, batch_idx):
        data = batch["data"]
        targets = batch["targets"]
        species_mask = batch["species_mask"]

        predictions = self.sigmoid_activation(self.model(data))

        if self.class_indices_to_eval is not None:
            predictions = predictions[:, self.class_indices_to_eval]
            targets = targets[:, self.class_indices_to_eval]

        self.log_metrics(mode="test", pred=predictions, y=targets, mask=species_mask)
