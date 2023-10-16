'''Setup: Sun-Huber type of dispersion function.'''

# External modules.
import numpy as np
import torch


###############################################################################


def rho(x):
    return np.sqrt(1.0+x**2)-1.0

def rho_torch(x):
    return torch.sqrt(1.0+x**2)-1.0

def rho_d1(x):
    return x / np.sqrt(1.0+x**2)

def rho_d1_torch(x):
    return x / torch.sqrt(1.0+x**2)

def rho_d2(x):
    return 1.0 / (1.0 + x**2)**(1.5)

def rho_d2_torch(x):
    return 1.0 / (1.0 + x**2)**(1.5)


###############################################################################
