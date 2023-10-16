'''Setup: various handy utilities.'''

# External modules.
import numpy as np
import torch
import os


###############################################################################


def do_normalization_per_feature(X):
    maxes = X.max(axis=0, keepdims=True)
    mins = X.min(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        X = (X-mins) / (maxes-mins)
        X[X == np.inf] = 0.0
        X = np.nan_to_num(X)
    return X


def do_normalization(X):
    maxval = X.max()
    minval = X.min()
    with np.errstate(divide="ignore", invalid="ignore"):
        X = (X-minval) / (maxval-minval)
        X[X == np.inf] = 0.0
        X = np.nan_to_num(X)
    return X


def get_seeds(base_seed, num):
    rg = np.random.default_rng(base_seed)
    seeds = rg.integers(low=1, high=2**20, size=(num,))
    return seeds


def compute_grad_norm(param_groups, dev):
    
    # Get l2 norms for all relevant parameters.
    norms = []
    for param_group in param_groups:
        for param in param_group["params"]:
            if param.grad is None:
                continue
            else:
                norms += [
                    torch.linalg.vector_norm(x=param.grad, ord=2).to(dev)
                ]
    
    # Get the l2 norm for the concatenated form.
    if len(norms) > 0:
        return torch.linalg.vector_norm(x=torch.stack(norms), ord=2)
    else:
        raise ValueError("No param grads for which to compute norm for.")


def my_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def not_all_same(list_to_check: list) -> bool:
    if all([x == list_to_check[0] for x in list_to_check]):
        return False
    else:
        return True


def makedir_safe(dirname) -> None:
    '''
    A simple utility for making new directories
    after checking that they do not exist.
    '''
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return None


###############################################################################
