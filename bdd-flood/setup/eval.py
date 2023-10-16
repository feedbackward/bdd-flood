'''Setup: design of the basic evaluation routines used.'''

# External modules.
import torch

# Internal modules.
from setup.losses import get_named_loss


###############################################################################


def eval_model_norm(model):
    '''
    Compute the l2 norm of the (concatenated) model parameters.
    '''
    norms = []
    for param in model.parameters():
        if param is None:
            continue
        else:
            norms += [torch.linalg.vector_norm(x=param, ord=2)]
    if len(norms) > 0:
        return torch.linalg.vector_norm(x=torch.stack(norms), ord=2).item()
    else:
        raise ValueError("No parameters for which to compute norm for.")


def eval_grad_norm(model, data_loader, loss_fn):
    '''
    Compute the l2 norm of the (concatenated) gradients.
    (probably won't implement this; other studies don't)
    '''
    raise NotImplementedError


def eval_loss_acc(model, data_loader, loss_fn):
    '''
    Performance evaluation on data specified by a data loader,
    in terms of a loss function specified by the user, and
    typical classification accuracy.

    Key points:
    - Assume the data loaders are *full batch*, i.e., no need to
      use any sum-reducing losses (default mean is fine).
    - Accuracy calculations assume model(X) yields vector
      outputs with length equal to the number of classes.
    '''
    
    loss_sum = 0.0
    num_correct = 0
    num_samples = 0

    batch_count = len(data_loader)

    if batch_count != 1:
        raise ValueError("Batch count for eval is not 1. Full batch please!")
    else:
        for X, Y in data_loader:
            Y_hat = model(X)
            loss = loss_fn(Y_hat, Y).item()
            zeroones = torch.where(
                torch.argmax(Y_hat, 1) == Y, 1.0, 0.0
            )
            accuracy = torch.mean(zeroones).item()
    
    return (loss, accuracy)


def run_evals(model, data_loaders, loss_name, loss_paras):
    
    eval_dl_tr, eval_dl_va, eval_dl_te = data_loaders
    
    loss_fn = get_named_loss(loss_name=loss_name,
                             loss_paras=loss_paras,
                             reduction="mean")
    
    loss_tr, acc_tr = eval_loss_acc(model=model,
                                    data_loader=eval_dl_tr,
                                    loss_fn=loss_fn)
    loss_va, acc_va = eval_loss_acc(model=model,
                                    data_loader=eval_dl_va,
                                    loss_fn=loss_fn)
    loss_te, acc_te = eval_loss_acc(model=model,
                                    data_loader=eval_dl_te,
                                    loss_fn=loss_fn)

    model_norm = eval_model_norm(model=model)
    
    metrics = {"loss_tr": loss_tr,
               "loss_va": loss_va,
               "loss_te": loss_te,
               "acc_tr": acc_tr,
               "acc_va": acc_va,
               "acc_te": acc_te,
               "model_norm": model_norm}
    
    return metrics


###############################################################################
