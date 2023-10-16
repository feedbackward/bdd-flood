'''Setup: generate data, set it in a loader, and return the loader.'''

# External modules.
import numpy as np
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

# Internal modules.
from setup.cifar10 import CIFAR10
from setup.cifar100 import CIFAR100
from setup.fashionmnist import FashionMNIST
from setup.gaussian import MultipleGaussians
from setup.nslkdd import NSL_KDD
from setup.sinusoid import Sinusoid
from setup.spiral import Spiral
from setup.svhn import SVHN


###############################################################################


# Functions related to data loaders and generators.

def get_dataloader(dataset_name, dataset_paras, device):
    '''
    This function does the following three things.
    1. Generate data in numpy format.
    2. Convert that data into PyTorch (tensor) Dataset object.
    3. Initialize PyTorch loaders with this data, and return them.
    '''
    
    dp = dataset_paras
    
    # First get the data generators.
    data_tr, data_va, data_te = get_generator(dataset_name=dataset_name,
                                              dataset_paras=dp)
    
    # Generate data, and map from ndarray to tensor, sharing memory.
    X_tr, Y_tr = map(torch.from_numpy, data_tr())
    X_va, Y_va = map(torch.from_numpy, data_va())
    X_te, Y_te = map(torch.from_numpy, data_te())
    
    # Do a dtype check.
    print("dtypes (tr): {}, {}".format(X_tr.dtype, Y_tr.dtype))
    print("dtypes (va): {}, {}".format(X_va.dtype, Y_va.dtype))
    print("dtypes (te): {}, {}".format(X_te.dtype, Y_te.dtype))
    
    # Organize tensors into PyTorch dataset objects.
    Z_tr = TensorDataset(X_tr.to(device), Y_tr.to(device))
    Z_va = TensorDataset(X_va.to(device), Y_va.to(device))
    Z_te = TensorDataset(X_te.to(device), Y_te.to(device))
    
    # Prepare the loaders to be returned.
    dl_tr = DataLoader(Z_tr, batch_size=dp["bs_tr"], shuffle=True)
    eval_dl_tr = DataLoader(Z_tr, batch_size=len(X_tr), shuffle=False)
    eval_dl_va = DataLoader(Z_va, batch_size=len(X_va), shuffle=False)
    eval_dl_te = DataLoader(Z_te, batch_size=len(X_te), shuffle=False)

    return (dl_tr, eval_dl_tr, eval_dl_va, eval_dl_te)


def get_generator(dataset_name, dataset_paras):
    
    dp = dataset_paras
    rg = dp["rg"]

    if dataset_name in ["gaussian", "sinusoid", "spiral"]:
        d = dp["dimension"]
        n_tr = dp["n_tr"]
        n_va = dp["n_va"]
        n_te = dp["n_te"]
        label_noise = dp["label_noise"]
    
    # Prepare the data generators for the dataset specified.
    if dataset_name == "cifar10":
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te = CIFAR10(rg=rg)() # note the call
        data_tr = lambda : (X_tr, Y_tr)
        data_va = lambda : (X_va, Y_va)
        data_te = lambda : (X_te, Y_te)

    elif dataset_name == "cifar100":
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te = CIFAR100(rg=rg)() # note the call
        data_tr = lambda : (X_tr, Y_tr)
        data_va = lambda : (X_va, Y_va)
        data_te = lambda : (X_te, Y_te)

    elif dataset_name == "fashionmnist":
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te = FashionMNIST(rg=rg)() # note call
        data_tr = lambda : (X_tr, Y_tr)
        data_va = lambda : (X_va, Y_va)
        data_te = lambda : (X_te, Y_te)
        
    elif dataset_name == "gaussian":
        data_means = [np.full(shape=(d,), fill_value=0.0),
                      np.full(shape=(d,), fill_value=1.0)]
        data_covs = [np.eye(N=d)/10.0, np.eye(N=d)/10.0]
        data_sizes_tr = [n_tr//2, n_tr//2]
        data_sizes_va = [n_va//2, n_va//2]
        data_sizes_te = [n_te//2, n_te//2]
        data_noises = [label_noise, label_noise]
        data_tr = MultipleGaussians(rg=rg, means=data_means, covs=data_covs,
                                    sizes=data_sizes_tr, noises=data_noises)
        data_va = MultipleGaussians(rg=rg, means=data_means, covs=data_covs,
                                    sizes=data_sizes_va, noises=data_noises)
        data_te = MultipleGaussians(rg=rg, means=data_means, covs=data_covs,
                                    sizes=data_sizes_te, noises=data_noises)
        
    elif dataset_name == "nslkdd":
        X, Y = NSL_KDD(rg=rg, noise=label_noise)() # note the call
        data_tr = lambda : (X[0:n_tr,:], Y[0:n_tr])
        data_va = lambda : (X[n_tr:(n_tr+n_va),:],
                            Y[n_tr:(n_tr+n_va)])
        data_te = lambda : (X[(n_tr+n_va):(n_tr+n_va+n_te),:],
                            Y[(n_tr+n_va):(n_tr+n_va+n_te)])
        
    elif dataset_name == "sinusoid":
        data_weight_pair = [np.array([1.0,1.0], dtype=np.float32),
                            np.array([-1.0,1.0], dtype=np.float32)]
        data_tr = Sinusoid(rg=rg, weight_pair=data_weight_pair,
                           size=n_tr, noise=label_noise)
        data_va = Sinusoid(rg=rg, weight_pair=data_weight_pair,
                           size=n_va, noise=label_noise)
        data_te = Sinusoid(rg=rg, weight_pair=data_weight_pair,
                           size=n_te, noise=label_noise)
        
    elif dataset_name == "spiral":
        data_sizes_tr = [n_tr//2, n_tr//2]
        data_sizes_va = [n_va//2, n_va//2]
        data_sizes_te = [n_te//2, n_te//2]
        data_tr = Spiral(rg=rg, sizes=data_sizes_tr, noise=label_noise)
        data_va = Spiral(rg=rg, sizes=data_sizes_va, noise=label_noise)
        data_te = Spiral(rg=rg, sizes=data_sizes_te, noise=label_noise)

    elif dataset_name == "svhn":
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te = SVHN(rg=rg)() # note the call
        data_tr = lambda : (X_tr, Y_tr)
        data_va = lambda : (X_va, Y_va)
        data_te = lambda : (X_te, Y_te)
        
    else:
        raise ValueError("Unrecognized dataset name.")
    
    return (data_tr, data_va, data_te)


###############################################################################
