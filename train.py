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

import src.dataloaders.dataloader as dataloader
import src.trainers.ctran_trainer as CtranTrainer
from src.trainers.mlp_trainer import MLPTrainer
from src.utils import load_opts


@hydra.main(config_path="configs", config_name="hydra")
def main(opts):
    hydra_opts = dict(OmegaConf.to_container(opts))
    args = hydra_opts.pop("args", None)

    base_dir = args["base_dir"]
    run_id = args["run_id"]
    if not base_dir:
        base_dir = get_original_cwd()

    config_path = os.path.join(base_dir, args["config"])
    default_config = os.path.join(base_dir, "configs/defaults.yaml")

    config = load_opts(config_path, default=default_config, commandline_opts=hydra_opts)
    global_seed = (run_id * (config.program.seed + (run_id - 1))) % (2**31 - 1)

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

    if config.Ctran.use:
        task = CtranTrainer.CTranTrainer(config)
    else:
        task = MLPTrainer(config)

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

    checkpoint_callback = ModelCheckpoint(
        monitor="val_topk",
        dirpath=config.save_path,
        save_top_k=1,
        mode="max",
        save_last=True,
        save_weights_only=True,
        auto_insert_metric_name=True,
    )

    trainer_args["callbacks"] = [checkpoint_callback]
    trainer_args["max_epochs"] = config.max_epochs
    trainer_args["check_val_every_n_epoch"] = 4
    trainer_args["accelerator"] = "gpu"

    trainer = pl.Trainer(**trainer_args)

    # Run experiment
    trainer.fit(model=task, datamodule=datamodule)
    trainer.test(model=task, datamodule=datamodule)

    # logging the best checkpoint to comet ML
    print(checkpoint_callback.best_model_path)
    trainer.logger.experiment.log_asset(
        checkpoint_callback.best_model_path, file_name="best_checkpoint.ckpt"
    )


if __name__ == "__main__":
    main()
