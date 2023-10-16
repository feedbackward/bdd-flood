'''Setup: model design and functions for getting specific models.'''

# External modules.
import torch.nn as nn
import torchvision


###############################################################################


class IshidaMLP_synthetic(nn.Module):
    '''
    The multi-layer perceptron used for synthetic data tests by
    Ishida et al. (2020).

    NOTE: they call this a "five hidden layer" MLP.
          Follows their public code exactly.
    '''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        width = 500
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, width, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, width, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, width, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, width, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, output_dim, bias=True)
        )
        return None

    def forward(self, x):
        return self.linear_relu_stack(x)


class IshidaMLP_benchmark(nn.Module):
    '''
    The multi-layer perceptron used for benchmark data tests
    in Ishida et al. (2020).

    NOTE: they call this a "two hidden layer" MLP.
          No public code, so we just infer based on
          prev example in IshidaMLP_synthetic.
          The BatchNorm1d inclusion is also inferred.
    '''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        width = 1000
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, width, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, output_dim, bias=True)
        )
        return None

    def forward(self, x):
        return self.linear_relu_stack(x)


def get_model(model_name, model_paras):
    
    mp = model_paras
    
    if model_name == "IshidaMLP_synthetic":
        return IshidaMLP_synthetic(input_dim=mp["dimension"],
                                   output_dim=mp["num_classes"])
    if model_name == "IshidaMLP_benchmark":
        return IshidaMLP_benchmark(input_dim=mp["dimension"],
                                   output_dim=mp["num_classes"])
    elif model_name == "ResNet18":
        return torchvision.models.resnet18(num_classes=mp["num_classes"])
    elif model_name == "ResNet34":
        return torchvision.models.resnet34(num_classes=mp["num_classes"])
    else:
        raise ValueError("Unrecognized model name.")


###############################################################################
