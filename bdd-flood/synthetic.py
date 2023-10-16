'''Main script: Synthetic data tests following Ishida et al. (2020).'''

# External modules.
from argparse import ArgumentParser
import mlflow
import numpy as np
import torch

# Internal modules.
from setup.data import get_dataloader
from setup.eval import run_evals
from setup.losses import get_loss
from setup.models import get_model
from setup.optimizers import get_optimizer
from setup.training import do_training


###############################################################################


def get_parser():
    parser = ArgumentParser(
        prog="synthetic",
        description="Synthetic data tests following Ishida et al. (2020).",
        add_help=True
    )
    parser.add_argument("--adaptive",
                        help="Set to 'yes' for adaptive SAM.",
                        type=str)
    parser.add_argument("--base-gpu-id",
                        default=0,
                        help="Specify which GPU should be the base GPU.",
                        type=int)
    parser.add_argument("--bs-tr",
                        help="Batch size for training data loader.",
                        type=int)
    parser.add_argument("--dataset",
                        help="Dataset name.",
                        type=str)
    parser.add_argument("--dimension",
                        help="Dimension of data space.",
                        type=int)
    parser.add_argument("--epochs",
                        help="Number of epochs in training loop.",
                        type=int)
    parser.add_argument("--eta",
                        help="Weight parameter on theta in SoftAD.",
                        type=float)
    parser.add_argument("--flood-level",
                        help="Flood level parameter for Ishida method.",
                        type=float)
    parser.add_argument("--force-cpu",
                        help="Either yes or no; parse within main().",
                        type=str)
    parser.add_argument("--force-one-gpu",
                        help="Either yes or no; parse within main().",
                        type=str)
    parser.add_argument("--label-noise",
                        help="Fraction of labels to randomly flip.",
                        type=float)
    parser.add_argument("--loss",
                        help="Name of the base loss function.",
                        type=str)
    parser.add_argument("--method",
                        help="Abstract method name.",
                        type=str)
    parser.add_argument("--model",
                        help="Model name.",
                        type=str)
    parser.add_argument("--momentum",
                        help="Momentum parameter for optimizers.",
                        type=float)
    parser.add_argument("--num-classes",
                        help="Number of classes (for classification tasks).",
                        type=int)
    parser.add_argument("--n-te",
                        help="Sample size of test data.",
                        type=int)
    parser.add_argument("--n-tr",
                        help="Sample size of training data.",
                        type=int)
    parser.add_argument("--n-va",
                        help="Sample size of validation data.",
                        type=int)
    parser.add_argument("--optimizer",
                        help="Optimizer name.",
                        type=str)
    parser.add_argument("--optimizer-base",
                        help="Base optimizer name (only for SAM).",
                        type=str)
    parser.add_argument("--radius",
                        help="Radius parameter for SAM method.",
                        type=float)
    parser.add_argument("--random-seed",
                        help="Integer-valued random seed.",
                        type=int)
    parser.add_argument("--sigma",
                        help="Scaling parameter for SoftAD.",
                        type=float)
    parser.add_argument("--step-size",
                        help="Step size parameter for optimizers.",
                        type=float)
    parser.add_argument("--theta",
                        help="Shift parameter for SoftAD.",
                        type=float)
    parser.add_argument("--weight-decay",
                        help="Weight decay parameter for optimizers.",
                        type=float)
    return parser


def get_args():
    parser = get_parser()
    return parser.parse_args()


def main(args):

    # Organize clerical arguments.
    force_cpu = True if args.force_cpu == "yes" else False
    force_one_gpu = True if args.force_one_gpu == "yes" else False
    base_gpu_id = args.base_gpu_id

    # Device setup.
    if force_cpu and not force_one_gpu:
        device = torch.device(type="cpu")
    elif force_one_gpu and not force_cpu:
        device = torch.device(type="cuda", index=base_gpu_id)
    else:
        raise ValueError("Please specify either CPU or single GPU setting.")
    
    # Seed the random generator (numpy and torch).
    rg = np.random.default_rng(args.random_seed)
    rg_torch = torch.manual_seed(seed=args.random_seed)
    
    # Get the data (placed on desired device).
    dataset_paras = {
        "rg": rg,
        "dimension": args.dimension,
        "bs_tr": args.bs_tr,
        "n_tr": args.n_tr,
        "n_va": args.n_va,
        "n_te": args.n_te,
        "label_noise": args.label_noise
    }
    dl_tr, eval_dl_tr, eval_dl_va, eval_dl_te = get_dataloader(
        dataset_name=args.dataset,
        dataset_paras=dataset_paras,
        device=device
    )
    
    # Initialize the model (placed on desired device).
    model_paras = {
        "rg": rg,
        "dimension": args.dimension,
        "num_classes": args.num_classes
    }
    model = get_model(model_name=args.model, model_paras=model_paras)
    print("Model:", model)
    model = model.to(device)
    
    # Set up the optimizer.
    opt_paras = {"momentum": args.momentum,
                 "step_size": args.step_size,
                 "weight_decay": args.weight_decay,
                 "adaptive": args.adaptive,
                 "radius": args.radius,
                 "optimizer_base": args.optimizer_base}
    optimizer = get_optimizer(opt_name=args.optimizer,
                              model=model,
                              opt_paras=opt_paras)
    print("Optimizer:", optimizer)

    # Get the loss function ready.
    loss_paras = {"method": args.method,
                  "flood_level": args.flood_level,
                  "theta": args.theta,
                  "sigma": args.sigma,
                  "eta": args.eta}
    loss_fn = get_loss(loss_name=args.loss,
                       loss_paras=loss_paras)
    print("loss_fn:", loss_fn)
    
    # Execute the training loop.
    for epoch in range(-1, args.epochs):

        print("Epoch: {}".format(epoch))
        
        # Do training step, except at initial epoch.
        if epoch >= 0:
            do_training(method=args.method,
                        model=model,
                        optimizer=optimizer,
                        dl_tr=dl_tr,
                        loss_fn=loss_fn)
        
        # Evaluation step.
        model.eval()
        with torch.no_grad():
            metrics = run_evals(
                model=model,
                data_loaders=(eval_dl_tr, eval_dl_va, eval_dl_te),
                loss_name=args.loss,
                loss_paras=loss_paras
            )
        
        # Log the metrics of interest.
        mlflow.log_metrics(step=epoch+1, metrics=metrics)

    # Finished.
    return None


if __name__ == "__main__":
    args = get_args()
    main(args)


###############################################################################
