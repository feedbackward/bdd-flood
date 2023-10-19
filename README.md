# bdd-flood: links between bi-directional dispersion and flooding

In this repository, we provide software and demonstrations related to the following paper:

- [Implicit regularization via soft ascent-descent](https://arxiv.org/abs/2310.10006). Matthew J. Holland and Kosuke Nakatani. *Preprint*.

This repository contains code which can be used to faithfully reproduce all the experimental results given in the above paper, and it can be easily applied to more general machine learning tasks outside the examples considered here.

## Our setup

Below, we describe the exact procedure by which we set up a (conda-based) virtual environment and installed the software used for our numerical experiments.

First, we [install miniforge](https://github.com/conda-forge/miniforge) (i.e., a conda-forge aligned miniconda).

Next, [installing PyTorch](https://pytorch.org/) after a bunch of other software can lead to some complications, and so I elected to install it first, as below.

```
$ conda create -n bdd-flood pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Note that the version of CUDA (here, 11.7) may need to be adjusted for your system. Now for JupyterLab and matplotlib, among others.

```
$ conda activate bdd-flood
(bdd-flood) $ conda install jupyterlab
```

This was super fast, no problem at all. The next command, however, was extremely slow.

```
(bdd-flood) $ conda install matplotlib
```

This took about 10 minutes, but in the end it did go through, and both torch and matplotlib can be imported and used together. Great.

Next, let's [use pip](https://pypi.org/project/pip/) to get [mlflow](https://mlflow.org/).

```
(bdd-flood) $ pip install mlflow
```

With proper scripts, we can easily access the result of experiments (= one or more runs) via the mlflow ui, as follows.

```
(bdd-flood) $ mlflow ui
```

This can be executed from the same directory as we execute the top driver scripts. That said, all the main figures in the paper are produced with Jupyter notebooks, not using the mlflow interface directly.


## Getting started

Anyone visiting this repository is probably interested in the software used to obtain the results shown in our paper. Any files not described explicitly are helper files used in support of the experiments carried out in the files given below.

### Simple demonstrations

There are a handful of Jupyter notebooks just dedicated to basic empirical investigations of the properties of SoftAD and Flooding, among other things. We list these notebooks here.

- `check_sims_*.ipynb`: these notebooks generate a sample of the data used in our synthetic experiments, and plot it.
- `demo_SoftAD.ipynb`: all the basic tests comparing SoftAD and Flooding (introduced in section 3 of the paper) are carried out within this notebook.
- `rho_smoothed.ipynb`: here we show how a simple scaling factor can smoothly bridge the gap between our soft threshold and the hard Flooding threshold.


### Main empirical tests

The core content of our empirical investigations (section 5 of the paper) is a pair of experiments, one using synthetic data (section 5.1), and one using real-world benchmarks (section 5.2).

Before we give the commands to execute each experiment, let us note that the key experiment settings are all specified in the following two files: `run_synthetic.py`, `run_benchmarks.py`. Modifying the experimental settings (e.g., number of trials, seeds, mini-batch size, hyperparameter grids, etc.) can be done completely within these two files. If, however, the user wants to modify the design of the experiment, in addition to modifying these two files, the mlflow project file `MLproject` will need to be modified, as will be the argument parsing done in the driver scripts `synthetic.py` and `benchmarks.py`.


#### Non-linear binary classification on the plane

Using our default settings, simply run the following command.

```
(bdd-flood) $ python run_synthetic.py
```

All the experimental settings are specified within this file `run_synthetic.py`, and the main driver script is `synthetic.py`. Results can be viewed using the Jupyter notebook called `eval_synthetic.ipynb`.


#### Image classification from scratch

Running these experiments is essentially analogous to those just described. Run the following command.

```
(bdd-flood) $ python run_benchmarks.py
```

All the experimental settings are specified within this file `run_benchmarks.py`, and the main driver script is `benchmarks.py`. Results can be viewed using the Jupyter notebook called `eval_benchmarks.ipynb`.


## Reference links

Introduction and documentation for conda-forge project.
https://conda-forge.org/docs/user/introduction.html

Uninstallation of conda-related materials.
https://docs.anaconda.com/free/anaconda/install/uninstall/

Getting miniforge.
https://github.com/conda-forge/miniforge

JupyterLab home page.
https://jupyter.org/

PyTorch, local installation.
https://pytorch.org/get-started/locally/
