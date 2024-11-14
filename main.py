# right now, used for splot-only training and testing.
import argparse
import logging
import os

import torch
import yaml
from pydantic import ValidationError
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

from src.config import Config
from src.trainers.splot_trainer import sPlotTrainer
from src.dataloaders.splot_dataloader import sPlotDataModule
from src.utils import save_test_results_to_csv


def load_config(config_path) -> Config:
    print(config_path)
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return Config(**config_dict)


def map_dataset_name_from_config(config):
    if config.dataset_name == "sPlot":
        return sPlotTrainer, sPlotDataModule
    else:
        raise ValueError(f"Dataset {config.dataset_name} not supported")


def main():
    # Argument parser for config file path
    parser = argparse.ArgumentParser(
        description="PyTorch Lightning Tabular Data MLP Training"
    )
    parser.add_argument("--config", type=str, default="config.yaml", required=True)
    args = parser.parse_args()

    try:
        # Load and validate configuration
        config = load_config(os.path.join(os.getcwd(), args.config))
    except ValidationError as e:
        print("Configuration validation error:", e.json())
        exit(1)

    print("Mode:", config.mode)
    print("Model Config:", config.model)
    print("Data Path Config:", config.data)
    print("Training Config:", config.training)

    # Create data module and trainer
    trainer_class, data_module_class = map_dataset_name_from_config(config)
    data_module = data_module_class(config.data)
    task = trainer_class(config)

    seed_everything(config.training.seed)

    # Initialize Comet.ml logger
    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        project_name=config.logger.project_name,
        experiment_name=config.logger.experiment_name,
    )
    os.makedirs(
        os.path.join(config.logger.checkpoint_path, config.logger.experiment_name),
        exist_ok=True,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_auroc",
        dirpath=os.path.join(
            config.logger.checkpoint_path, config.logger.experiment_name
        ),
        save_top_k=1,
        every_n_epochs=2,
        mode="max",
        save_last=True,
        auto_insert_metric_name=True,
    )

    # Train the model
    trainer = Trainer(
        max_epochs=config.training.max_epochs,
        logger=comet_logger,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        callbacks=[checkpoint_callback],
    )

    if config.mode == "train":
        trainer.fit(model=task, datamodule=data_module)
        trainer.test(model=task, datamodule=data_module)
    else:
        checkpoint_path = os.path.join(
                    config.logger.checkpoint_path,
                    config.logger.experiment_name,
                    config.logger.checkpoint_name)
        logging.info("loading checkpoint %s", checkpoint_path)
        task.load_state_dict(
            torch.load(checkpoint_path)["state_dict"],
        )
        test_results = trainer.test(model=task, datamodule=data_module, verbose=True)
        logging.info("test results: %s", test_results)
        save_test_results_to_csv(
            results=test_results[0],
            root_dir=os.path.join(config.logger.checkpoint_path, config.logger.experiment_name),
            file_name="test_results.csv",
        )


if __name__ == "__main__":
    main()
