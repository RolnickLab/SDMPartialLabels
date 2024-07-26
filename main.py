import argparse
import logging
import os

import yaml
from pydantic import ValidationError
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

from new_src.config import Config
from new_src.dataloader import TabularDataModule
from new_src.trainer import sPlotsTrainer


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return Config(**config_dict)


def main():
    # Argument parser for config file path
    parser = argparse.ArgumentParser(
        description="PyTorch Lightning Tabular Data MLP Training"
    )
    parser.add_argument("--config", type=str, default="config.yaml", required=True)
    args = parser.parse_args()

    try:
        # Load and validate configuration
        config = load_config(args.config)
    except ValidationError as e:
        print("Configuration validation error:", e.json())
        exit(1)

    print("Mode:", config.mode)
    print("Model Config:", config.model)
    print("Data Path Config:", config.data)
    print("Training Config:", config.training)

    # Create data module
    data_module = TabularDataModule(config.data, batch_size=config.training.batch_size)
    task = sPlotsTrainer(config)

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
        monitor="val_accuracy",
        dirpath=os.path.join(
            config.logger.checkpoint_path, config.logger.experiment_name
        ),
        save_top_k=1,
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
        trainer.fit(task, data_module)
    else:
        task = task.load_from_checkpoint(
            task=task,
            checkpoint_path=os.path.join(
                config.logger.checkpoint_path,
                config.logger.experiment_name,
                "last.ckpt",
            ),
        )

        val_results = trainer.validate(model=task, datamodule=data_module)
        test_results = trainer.test(
            model=task, dataloaders=data_module.test_dataloader(), verbose=True
        )
        logging.info("validation results: %s", val_results)
        logging.info("test results: %s", test_results)


if __name__ == "__main__":
    main()
