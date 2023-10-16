'''Setup: initialize optimizer objects with desired settings.'''

# External modules.
import torch
from torch.optim import SGD, Adam, Optimizer

# Internal modules.
from setup.utils import compute_grad_norm


###############################################################################


# Optimizer-related parsers.

base_optimizers = ["SGD", "Adam"]

def get_base_optimizer(opt_name, opt_paras, model=None, params=None):
    '''
    Usage: pass either a model OR a parameter iterable, not both.
    '''
    if model is None:
        params = params
    elif params is None:
        params = model.parameters()
    else:
        raise ValueError("Pass model or params, not both.")
    
    op = opt_paras
    if opt_name == "SGD":
        optimizer = SGD(
            params,
            weight_decay=op["weight_decay"],
            lr=op["step_size"],
            momentum=op["momentum"]
        )
    elif opt_name == "Adam":
        optimizer = Adam(
            params,
            weight_decay=op["weight_decay"],
            lr=op["step_size"]
        )
    else:
        raise ValueError("Unrecognized optimizer name.")
    
    return optimizer


def get_optimizer(opt_name, opt_paras, model):
    op = opt_paras
    if opt_name in base_optimizers:
        optimizer = get_base_optimizer(opt_name=opt_name,
                                       opt_paras=opt_paras,
                                       model=model)
    elif opt_name == "SAM":
        opt_base_name = op.pop("optimizer_base")
        radius = op.pop("radius")
        adaptive = op.pop("adaptive")
        optimizer = SAM(
            params=model.parameters(),
            opt_base_name=opt_base_name,
            opt_base_paras=op,
            radius=radius,
            adaptive=adaptive
        )
    else:
        raise ValueError("Unrecognized optimizer name.")
    
    return optimizer


def get_constructor(opt_name):
    if opt_name == "SGD":
        return SGD
    elif opt_name == "Adam":
        return Adam
    else:
        raise ValueError("Unrecognized optimizer name.")


# Specialized optimizer implementations.

class SAM(Optimizer):
    '''
    Sharpness-aware minimization (Foret et al., 2021).
    Based on the following implementation:
    https://github.com/davda54/sam/blob/main/sam.py
    (This repo is linked in original SAM paper)

    - params: model parameters to be optimized.
    - opt_base_name: name for base optimizer.
    - radius: gradient norm size is fixed to this.
    - adaptive: ?
    - base_paras: options to be sent to the base optimizer constructor.
    '''
    
    def __init__(self, params, opt_base_name,
                 opt_base_paras, radius, adaptive):
        
        # Convert to boolean for convenience.
        # (still doesn't do anything though!)
        self.adaptive = True if adaptive == "yes" else False
        
        # The usual name for optimizer option dict is "defaults".
        defaults = {
            "radius": radius,
            "adaptive": adaptive,
            **opt_base_paras
        }
        
        # Initialize as vanilla Optimizer object with param_groups attribute.
        # Note that param_groups is always a list of dictionaries, and
        # Tensor-like Parameter objects to be optimized are stored (as lists)
        # under the key "params" within each param_group dict; the other
        # options are stored in each dict with their keys as-is.
        super().__init__(params, defaults)
        
        # Construct the base optimizer. It is okay to pass a list of typical
        # Parameter objects OR a list of "parameter groups" (dicts), and since
        # the preceding call of super() ensures we have such groups, it is
        # natural to pass this along.
        self.optimizer_base = get_base_optimizer(opt_name=opt_base_name,
                                                 opt_paras=opt_base_paras,
                                                 params=self.param_groups)
        
        # Update in case any new parameter groups have been added by the
        # base optimizer constructor.
        self.param_groups = self.optimizer_base.param_groups
        
        # Update options to include default values used by the base constructor
        # that were not specified using base_paras.
        self.defaults.update(self.optimizer_base.defaults)
        
        return None
    
    
    def first_step(self):
        '''
        Store the old parameter, and compute the perturbed
        variant which will be used in second step update.
        '''
        grad_norm = self.grad_norm()
        
        for param_group in self.param_groups:

            scale = param_group["radius"] / (grad_norm + 1e-12)

            for param in param_group["params"]:

                if param.grad is None:
                    # If has no gradient, there is nothing to do.
                    continue
                else:
                    # Use the handy state defaultdict to store old param.
                    self.state[param]["old_param"] = param.data.clone()

                    # Update the parameter by perturbation.
                    param.add_(param.grad, alpha=scale)
                    #param.add_(param.grad * scale.to(param)) # alternative.

        return None
    
    
    def second_step(self):
        '''
        After having run the first step and re-evaluating the
        objective function to get a fresh set of gradients at
        the perturbed point, we just need to use this to update.
        '''
        for param_group in self.param_groups:
            for param in param_group["params"]:
                if param.grad is None:
                    continue
                else:
                    # Go back to initial point for updating.
                    param.data = self.state[param]["old_param"]
        
        # Base optimizer update, noting grads are at the perturbed point.
        self.optimizer_base.step()
        
        return None

    
    def step(self, closure=None):
        raise NotImplementedError
    
    
    def grad_norm(self):

        # Put everything on same device in case of model parallelism.
        dev = self.param_groups[0]["params"][0].device

        # Get l2 norm for concatenated parameters.
        norm = compute_grad_norm(param_groups=self.param_groups, dev=dev)
        return norm
    
    
    def load_state_dict(self, state_dict):
        '''
        Exactly as defined in davda54's SAM repository.
        '''
        super().load_state_dict(state_dict)
        self.optimizer_base.param_groups = self.param_groups


###############################################################################
