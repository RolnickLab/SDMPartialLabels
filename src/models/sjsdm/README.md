## Getthing started with sjSDM

We use the implementation of sjSDM at [https://github.com/TheoreticalEcology/s-jSDM](https://github.com/TheoreticalEcology/s-jSDM) using the Python implementation. 

We install the following requirements: 
```filelock          3.20.0
fsspec            2025.10.0
Jinja2            3.1.6
madgrad           1.3
MarkupSafe        3.0.3
mpmath            1.3.0
networkx          3.6
numpy             2.3.5
opt_einsum        3.4.0
pillow            12.0.0
pip               24.2
pyro-api          0.1.2
pyro-ppl          1.9.1
pytorch-ranger    0.1.1
setuptools        80.9.0
sympy             1.14.0
torch             2.9.1
torch-optimizer   0.3.0
torchvision       0.24.1
tqdm              4.67.1
typing_extensions 4.15.0
```
and run the script inside the `s-jSDM/sjSDM/inst/python/` folder. 

In order to run sjSDM, we prepare the environmental data in a num_sites x num_covariates matrix (`env.npy` in the script) which was created following the script in `utils.py`. 

The code the train an sjSDM on sPlotOpen and run inference in the conditioned and unconditioned case is in `sjsdm-splot.py`. 
