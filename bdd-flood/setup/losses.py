'''Setup: initialize and pass the desired loss function.'''

# External modules.
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss, BCELoss

# Internal modules.
from setup.sunhuber import rho_torch


###############################################################################


# Here we define some customized loss classes.

class Loss_Flood(Module):
    '''
    General purpose loss class for the "flooding" algorithm
    of Ishida et al. (2020).
    '''
    def __init__(self, flood_level: float, loss_name: str, loss_paras: dict):
        super().__init__()
        self.flood_level = flood_level
        self.loss_fn = get_named_loss(loss_name=loss_name,
                                      loss_paras=loss_paras,
                                      reduction="mean")
        return None

    def forward(self, input: Tensor, target: Tensor):
        fl = self.flood_level
        loss = self.loss_fn(input=input, target=target)
        return (loss-fl).abs()+fl


class Loss_SoftAD(Module):
    '''
    Soft ascent-descent (Soft-AD), our most basic modified version of
    the flooding algorithm.
    '''
    def __init__(self, theta: float, sigma: float, eta: float,
                 loss_name: str, loss_paras: dict):
        super().__init__()
        self.theta = theta
        self.sigma = sigma
        self.eta = eta
        self.loss_fn = get_named_loss(loss_name=loss_name,
                                      loss_paras=loss_paras,
                                      reduction="none")
        return None
    
    def forward(self, input: Tensor, target: Tensor):
        theta = self.theta
        sigma = self.sigma + 1e-12 # to be safe.
        eta = self.eta
        loss = self.loss_fn(input=input, target=target)
        dispersion = (sigma**2) * rho_torch((loss-theta)/sigma).mean()
        return eta*theta + dispersion


class Loss_iFlood(Module):
    '''
    iFlood by Xie et al. (2022).
    '''
    def __init__(self, theta: float, sigma: float, eta: float,
                 loss_name: str, loss_paras: dict):
        super().__init__()
        self.theta = theta
        self.sigma = sigma
        self.eta = eta
        self.loss_fn = get_named_loss(loss_name=loss_name,
                                      loss_paras=loss_paras,
                                      reduction="none")
        return None
    
    def forward(self, input: Tensor, target: Tensor):
        theta = self.theta
        sigma = self.sigma + 1e-12 # to be safe.
        eta = self.eta
        loss = self.loss_fn(input=input, target=target)
        dispersion = (sigma**2) * ((loss-theta)/sigma).abs().mean()
        return eta*theta + dispersion


# Here we define various loss function getters.

def get_loss(loss_name, loss_paras):
    '''
    Loss function getter for all methods.
    '''
    ln = loss_name
    lp = loss_paras
    
    if lp["method"] == "ERM":
        loss_fn = get_named_loss(loss_name=ln,
                                 loss_paras=lp,
                                 reduction="mean")
    elif lp["method"] == "Ishida":
        loss_fn = get_flood_loss(loss_name=ln,
                                 loss_paras=lp)
    elif lp["method"] == "SAM":
        loss_fn = get_named_loss(loss_name=ln,
                                 loss_paras=lp,
                                 reduction="mean")
    elif lp["method"] == "SoftAD":
        loss_fn = get_softad_loss(loss_name=ln,
                                  loss_paras=lp)
    elif lp["method"] == "iFlood":
        loss_fn = get_iflood_loss(loss_name=ln,
                                  loss_paras=lp)
    else:
        raise ValueError("Unrecognized method name.")
    
    return loss_fn


def get_flood_loss(loss_name, loss_paras):
    '''
    A simple wrapper for Loss_Flood.
    '''
    fl = loss_paras["flood_level"]
    loss_fn = Loss_Flood(flood_level=fl,
                         loss_name=loss_name,
                         loss_paras=loss_paras)
    return loss_fn


def get_softad_loss(loss_name, loss_paras):
    '''
    A simple wrapper for Loss_SoftAD.
    '''
    eta = loss_paras["eta"]
    sigma = loss_paras["sigma"]
    theta = loss_paras["theta"]
    loss_fn = Loss_SoftAD(theta=theta, sigma=sigma, eta=eta,
                          loss_name=loss_name, loss_paras={})
    return loss_fn


def get_iflood_loss(loss_name, loss_paras):
    '''
    A simple wrapper for Loss_iFlood.
    '''
    eta = loss_paras["eta"]
    sigma = loss_paras["sigma"]
    theta = loss_paras["theta"]
    loss_fn = Loss_iFlood(theta=theta, sigma=sigma, eta=eta,
                          loss_name=loss_name, loss_paras={})
    return loss_fn


def get_named_loss(loss_name, loss_paras, reduction):
    
    if loss_name == "CrossEntropy":
        loss_fn = CrossEntropyLoss(reduction=reduction)
    elif loss_name == "BCELoss":
        loss_fn = BCELoss(reduction=reduction)
    else:
        raise ValueError("Unrecognized loss name.")
    
    return loss_fn


###############################################################################
