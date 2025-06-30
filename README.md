# CISO-SDM

This repository contains the code to reproduce the results from the paper: **CISO - Species Distribution Modeling
Conditioned on Incomplete Species Observations**.

## Installation

Code runs on Python 3.11. You can install pip packages from `requirements/requirements.txt`

We recommend following these steps for installing the required packages:

```conda create -n ciso_env python=3.11```

```conda activate ciso_env```

```pip install -r requirements.txt```

## Datasets:

All datasets and data files are publicly released [here](https://huggingface.co/cisosdm/datasets).

#### Data preparation:

The folder `data_preprocessing` include preparation files for:

* SatButterfly dataset: `ebutterfly_data_preparation`
* sPlotOpen: `prepare_sPlotOpen_data.ipynb`
* SatBirdxsPlotOpen co-located data: `prepare_satbirdxsplots.ipynb`

## Experiment configurations:

The folder `configs` include one folder for each dataset setup:

* `configs/satbird`
* `configs/satbirdxsatbutterfly`
* `configs/satbirdxsplot`
* `configs/splot`

Under each folder, there exists a file for each model reported in our work. A config file supports both training and
evaluating a model:

* `config_ciso.yaml`: CISO model
* `config_linear.yaml`: linear model
* `config_maxent.yaml`: linear model with maxent features
* `config_mlp.yaml`: mlp model
* `config_mlp_plusplus.yaml`: MLP++ model

## Trained model checkpoints:

All model checkpoints are publicly released [here](https://huggingface.co/cisosdm/model_checkpoints).

## Running code:

### Training:

To log experiments on comet-ml, make sure you have exported your COMET_API_KEY and COMET_WORKSPACE in your environmental
variables.
You can do so with `export COMET_API_KEY=your_comet_api_key` in your terminal.

* To train the model: `python train.py args.config=configs/`. Examples of all config files for different models and
  datasets
  are available in `configs`.

### Evaluation:

We mainly use the parameter `predict_family_of_species` to control which family subset of species we are
evaluating. `predict_family_of_species` defaults to `-1` during training.

#### Species within a single taxonomy setup:
SatBird:
- `predict_family_of_species = 0` : evaluate non-songbirds
- `predict_family_of_species = 1` : evaluate songbirds

splot:
- `predict_family_of_species = 0` : evaluate non-trees
- `predict_family_of_species = 1` : evaluate trees

For models that support partial labels such as CISO and MLP++:

- To evaluate with **no partial labels** given (everything is unknown), set `eval_known_rate == 0 `
- To evaluate with **partial labels** given (other group labels known), set `eval_known_rate == 1 `

#### Species in Multi-taxa setup:
SatBird & SatButterfly:
- `predict_family_of_species = 0` : evaluate birds
- `predict_family_of_species = 1` : evaluate butterflies

SatBird & sPlotOpen:
- `predict_family_of_species = 0` : evaluate birds
- `predict_family_of_species = 1` : evaluate plants

For models that support partial labels such as CISO and MLP++:

- To evaluate with **no partial labels** given (everything is unknown), set `eval_known_rate == 0 `
- To evaluate with **partial labels** given (other group labels known), set `eval_known_rate == 1 `

## Reproducing Results:

## Reproducing Figures:



This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License](https://creativecommons.org/licenses/by-nc/4.0/).
