### Bayesian Inference and MCMC sampling

This section of the repo contains an introduction to Bayesian Inference and Markov Chain Monte Carlo (MCMC) sampling. Note that the LaTeX formulas in the jupyter notebook won't be displayed properly in GitHub. It is therefore best to clone the repo and use the jupyter notebook extensions. Please see the [top-level page](https://github.com/wutobias/CompSalad) of this repo for more details on how to install these.

* **Bayesian-Inference-toyexample.ipynb** This is the main piece of this section. It contains a structured introduction to Bayesian Inference and a toy example to experiment with.

* **MClearn.yml** This file can be used to set up a conda environment for this particular section. It is not necessary though, since the ipython notebook contains a line that installs (likely) missing packages. Nonetheless, to set up a new environment do the following: `conda env create -f MClearn.yml`

* **mcmc_sampler.py** This file contains commented classes and routines necessary for running the ipython notebook. It must be stored in the same folder as the notebook.
