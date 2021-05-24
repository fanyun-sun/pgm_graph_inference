# Inference in graphical models with GNNs

*Authors: Ksenia Korovina, Lingxiao Zhao, Mark Cheung, Wenwen Si*

##

* To generate data: `$ ./make_data.sh grid`
* To run experiments: `$./run_all.sh grid MODEL TRAIN_NUM`
* For the list of available models, refer to `inference/__init.py`

## Structure of the repo

* `graphical_models` directory contains definitions of `GraphicalModel` class (objects abbreviated as "graphs"); `graphical_models/datasets` contains labeled data. Labeled graphs are stored as `.npy` files in the following directory structure:
```
graphical_models/datasets/{train/val/test}
                                |-- star/
                                |    |-  9/<file1.npy>, <file2.npy> ...
                                |    |- 10/
                                     |- 11/
                               ...  ...
```
* `inference` directory contains all methods for performing inference on a `GraphicalModel`, including BP and GNN-based methods; `inference/pretrained` contains pretrained GNN inference procedures.
* `experiments` directory contains specification for loading data (combinations of graph structures and sizes to compose training and testing) and inference experiments. If an experiment uses GNN inference, it first checks if an appropriate pretrained model exists (using `train.py`) by matching model postfix and `train_set_name` in experiment.
* `create_data.py` generates graphs from user-specified arguments and saves to `graphical_models/datasets` by default.
* `train.py` uses one of generated datasets to train a GNN inference procedure (such as GatedGNN).

## Installing dependencies

The following command will install several python packages for graphs, numerical computations and deep learning:

```bash
pip install -r requirements.txt
```

Installation of `igraph` may fail under MacOS and anaconda. In this case, try setting `MACOSX_DEPLOYMENT_TARGET`:

```bash
MACOSX_DEPLOYMENT_TARGET=10.9 pip install graphkernels
```

## Getting started

For imports to work correctly, add root of the repository to `PYTHONPATH` by running

```bash
source setup.sh
```

## Usage

To generate data to reproduce experiments, run

```bash
bash prepare_tree_experiment.sh  # for main experiment 1
```

To train the GNN inference, use `train.py`. Finally, use `./experiments/run_exps.py` to specify the way to compare a trained GNN with other inference methods.


## References

[ICLR18](https://arxiv.org/abs/1803.07710)
