"""
main training script
To run: python train.py args.config=$CONFIG_FILE_PATH
"""

import os

import hydra
import pytorch_lightning as pl
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.utilities import rank_zero_info

import src.dataloaders.dataloader as dataloader
import src.trainers.sdm_partial_trainer as SDMPartialTrainer
from src.trainers.baseline_trainer import BaselineTrainer
from src.utils import load_opts


class MultiMetricCheckpoint(pl.callbacks.Callback):
    """
    A custom callback that only saves a checkpoint if BOTH:
      - 'val_topk_bird' is better (lower) than the best we've seen, AND
      - 'val_topk_butterfly' is better (higher) than the best we've seen.
    """

    def __init__(
            self,
            dirpath: str,
            filename: str,
            monitor_metric_1: str,
            monitor_metric_2: str,
            mode_metric: str = "max"
    ):
        super().__init__()
        os.makedirs(dirpath, exist_ok=True)

        # Initialize “best so far” for each metric
        self.best_metric_1 = -float("inf") if mode_metric == "max" else float("inf")
        self.best_metric_2 = -float("inf") if mode_metric == "max" else float("inf")

        self.monitor_metric_1 = monitor_metric_1
        self.monitor_metric_2 = monitor_metric_2
        self.mode_metric = mode_metric

        self.dirpath = dirpath
        self.filename = filename
        self.last_ckpt_path = None

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics = trainer.callback_metrics

        # Ensure both metrics are logged this step
        if self.monitor_metric_1 not in metrics or self.monitor_metric_2 not in metrics:
            return

        current_metric_1 = metrics[self.monitor_metric_1].item()
        current_metric_2 = metrics[self.monitor_metric_2].item()

        # Check whether each metric individually improved
        metric_1_improved = (
            current_metric_1 > self.best_metric_1 if self.mode_metric == "max"
            else current_metric_1 < self.best_metric_1
        )
        metric_2_improved = (
            current_metric_2 > self.best_metric_2 if self.mode_metric == "max"
            else current_metric_2 < self.best_metric_2
        )

        # Only save if BOTH metrics improved
        if metric_1_improved and metric_2_improved:
            if self.last_ckpt_path is not None and os.path.isfile(self.last_ckpt_path):
                try:
                    os.remove(self.last_ckpt_path)
                    rank_zero_info(f"Deleted previous checkpoint: {self.last_ckpt_path}")
                except Exception as e:
                    rank_zero_info(f"Warning: could not delete {self.last_ckpt_path}: {e}")

            # Update the best-so-far values
            self.best_metric_1 = current_metric_1
            self.best_metric_2 = current_metric_2

            # Build the checkpoint filepath
            filepath = os.path.join(
                self.dirpath,
                self.filename.format(
                    epoch=trainer.current_epoch,
                    **{self.monitor_metric_1: current_metric_1, self.monitor_metric_2: current_metric_2},
                ),
            )
            rank_zero_info(f"Saving new multi‐metric checkpoint: {filepath}")
            trainer.save_checkpoint(filepath)


@hydra.main(config_path="configs", config_name="hydra")
def main(opts):
    hydra_opts = dict(OmegaConf.to_container(opts))
    args = hydra_opts.pop("args", None)

    run_id = args["run_id"]

    base_dir = get_original_cwd()

    config_path = os.path.join(base_dir, args["config"])
    default_config = os.path.join(base_dir, "configs/defaults.yaml")

    config = load_opts(config_path, default=default_config, commandline_opts=hydra_opts)
    global_seed = (run_id * (config.training.seed + (run_id - 1))) % (2**31 - 1)

    # naming experiment folders with seed information
    config.save_path = os.path.join(base_dir, config.save_path, str(global_seed))
    config.comet.experiment_name = (
        config.comet.experiment_name + "_seed_" + str(global_seed)
    )
    config.base_dir = base_dir

    # set global seed
    pl.seed_everything(global_seed)
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    with open(os.path.join(config.save_path, "config.yaml"), "w") as fp:
        OmegaConf.save(config=config, f=fp)
    fp.close()

    datamodule = dataloader.SDMDataModule(config)

    if config.partial_labels.use:
        task = SDMPartialTrainer.SDMPartialTrainer(config)
    else:
        task = BaselineTrainer(config)

    trainer_args = {}

    if config.log_comet:
        if os.environ.get("COMET_API_KEY"):
            comet_logger = CometLogger(
                api_key=os.environ.get("COMET_API_KEY"),
                workspace=os.environ.get("COMET_WORKSPACE"),
                # save_dir=".",  # Optional
                project_name=config.comet.project_name,  # Optional
                experiment_name=config.comet.experiment_name,
            )
            comet_logger.experiment.add_tags(list(config.comet.tags))
            trainer_args["logger"] = comet_logger
        else:
            print("no COMET API Key found..continuing without logging..")
            return

    if config.data.multi_taxa:
        val_monitor_1 = config.data.monitor_metric_1
        val_monitor_2 = config.data.monitor_metric_2
        checkpoint_callback = MultiMetricCheckpoint(
            dirpath=config.save_path,
            filename=(
                f"best_epoch{{epoch:02d}}"
                f"-taxa1-{{{val_monitor_1}:.4f}}"
                f"-taxa2-{{{val_monitor_2}:.4f}}.ckpt"
            ),
            monitor_metric_1=val_monitor_1,
            monitor_metric_2=val_monitor_2,
            mode_metric="max",
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_topk",
            dirpath=config.save_path,
            save_top_k=1,
            filename="best-{epoch:02d}-{val_topk:.4f}",
            mode="max",
            save_last=True,
            save_weights_only=True,
            auto_insert_metric_name=True,
        )
    trainer_args["callbacks"] = [checkpoint_callback]
    trainer_args["max_epochs"] = config.training.max_epochs
    trainer_args["check_val_every_n_epoch"] = 4
    trainer_args["accelerator"] = config.training.accelerator

    trainer = pl.Trainer(**trainer_args)

    # Run experiment
    trainer.fit(model=task, datamodule=datamodule)
    trainer.test(model=task, datamodule=datamodule)


if __name__ == "__main__":
    main()
