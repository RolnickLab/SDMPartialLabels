"""
main testing script
To run: python test.py args.config=CONFIG_FILE_PATH
"""

import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from pytorch_lightning.loggers import CometLogger

import src.dataloaders.dataloader as dataloader
import src.trainers.sdm_partial_trainer as SDMPartialTrainer
from src.trainers.baseline_trainer import BaselineTrainer
from src.utils import load_opts, save_test_results_to_csv

hydra_config_path = Path(__file__).resolve().parent / "configs/hydra.yaml"


def load_existing_checkpoint(task, base_dir, checkpint_path):
    print("Loading existing checkpoint")
    task.load_state_dict(
        torch.load(os.path.join(base_dir, checkpint_path))["state_dict"]
    )

    return task


def get_seed(run_id, fixed_seed):
    return (run_id * (fixed_seed + (run_id - 1))) % (2**31 - 1)


@hydra.main(config_path="configs", config_name="hydra")
def main(opts):
    hydra_opts = dict(OmegaConf.to_container(opts))
    args = hydra_opts.pop("args", None)
    run_id = args["run_id"]

    base_dir = get_original_cwd()

    config_path = os.path.join(base_dir, args["config"])
    default_config = os.path.join(base_dir, "configs/defaults.yaml")

    config = load_opts(config_path, default=default_config, commandline_opts=hydra_opts)
    global_seed = (run_id * (config.training.seed + (run_id - 1))) % (2 ** 31 - 1)

    config.base_dir = base_dir
    config.save_path = os.path.join(base_dir, config.save_path, str(global_seed))
    config.load_ckpt_path = os.path.join(base_dir, config.load_ckpt_path)

    if "file_name" in args:
        config.file_name = args["file_name"]
    else: 
        config.file_name = "test_results.csv"

    datamodule = dataloader.SDMDataModule(config)
    datamodule.setup()
    if config.partial_labels.use:
        task = SDMPartialTrainer.SDMPartialTrainer(config)
    else:
        task = BaselineTrainer(config)

    trainer_args = {}
    trainer_args["accelerator"] = config.training.accelerator

    if config.comet.experiment_key:
        comet_logger = CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            workspace=os.environ.get("COMET_WORKSPACE"),
            project_name=config.comet.project_name,
            experiment_name=config.comet.experiment_name,
            experiment_key=config.comet.experiment_key,
        )

        trainer_args["logger"] = comet_logger

    def test_task(task):
        trainer = pl.Trainer(**trainer_args)
        test_results = trainer.test(model=task, datamodule=datamodule, verbose=True)
        print("Final test results: ", test_results)
        return test_results

    # if a single checkpoint is given
    if config.load_ckpt_path:
        if config.load_ckpt_path.endswith(".ckpt"):
            task = load_existing_checkpoint(
                task=task,
                base_dir="",
                checkpint_path=config.load_ckpt_path,
            )
            global_seed = get_seed(args["run_id"], config.training.seed)
            pl.seed_everything(global_seed)

            test_results = test_task(task)

            save_test_results_to_csv(
                results=test_results[0],
                root_dir=config.save_path,
                file_name=config.file_name
            )
        else:
            # get the number of experiments based on folders given
            
            n_runs = len([ f for f in os.scandir(config.load_ckpt_path) if f.is_dir() ])
            # loop over all seeds
            for run_id in range(1, n_runs + 1):
                # get path of a single experiment
                run_id_path = os.path.join(
                    config.load_ckpt_path, str(get_seed(run_id, config.training.seed))
                )
                global_seed = get_seed(run_id, config.training.seed)
                pl.seed_everything(global_seed)

                # get path of the best checkpoint (not last)
                files = os.listdir(os.path.join(config.base_dir, run_id_path))
                best_checkpoint_file_name = [
                    file
                    for file in files
                    if "last" not in file and file.endswith(".ckpt")
                ][0]
                print(best_checkpoint_file_name)
                checkpoint_path_per_run_id = os.path.join(
                    run_id_path, best_checkpoint_file_name
                )
                # load the best checkpoint for the given run
                task = load_existing_checkpoint(
                    task=task,
                    base_dir=config.base_dir,
                    checkpint_path=checkpoint_path_per_run_id,
                )
                test_results = test_task(task)
                save_test_results_to_csv(
                    results=test_results[0],
                    root_dir=config.save_path,
                    file_name=config.file_name
                )

    else:
        print("No checkpoint provided...Evaluating a random model")
        _ = test_task(task)


if __name__ == "__main__":
    main()
