# SDM with Partial Labels

### Evaluation:
We use the parameter `predict_family_of_species` to control which family subset of species we are evaluating
### Species within a single taxonomy setup:
SatBird:
- `predict_family_of_species = 0` : evaluate non-songbirds
- `predict_family_of_species = 1` : evaluate songbirds

splot:
- `predict_family_of_species = 0` : evaluate non-trees
- `predict_family_of_species = 1` : evaluate trees

### Species in Multi-taxa setup:
SatBird & SatButterfly:
- To evaluate with **no partial labels** given (everything is unknown), set `eval_known_rate == 0 `
- `predict_family_of_species = 0` : evaluate birds
- `predict_family_of_species = 1` : evaluate butterflies
	
- To evaluate with **partial labels** given (some labels known), set `eval_known_rate == 1 `
- `predict_family_of_species = 0` : evaluate birds
- `predict_family_of_species = 1` : evaluate butterflies


### Running code:

#### Installation 
Code runs on Python 3.10. You can create conda env using `requirements/environment.yaml` or install pip packages from `requirements/requirements.txt`

We recommend following these steps for installing the required packages: 

```conda env create -f requirements/environment.yaml``` 

```conda activate satbird```

```conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia```

#### Training and testing

* To train the model (check `run_files/job.sh`) : `python train.py args.config=configs/base.yaml`. Examples of all config files for different baselines 
are available in `configs`.
* To train a model: `python train.py args.config=$CONFIG_FILE_NAME `
* To test a model: `python test.py args.config=$CONFIG_FILE_NAME `

To log experiments on comet-ml, make sure you have exported your COMET_API_KEY and COMET_WORKSPACE in your environmental variables.
You can do so with `export COMET_API_KEY=your_comet_api_key` in your terminal.


This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License](https://creativecommons.org/licenses/by-nc/4.0/).
