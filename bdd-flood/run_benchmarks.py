'''Top script: Benchmark data tests following Ishida et al. (2020).'''

# External modules.
import copy
import mlflow
import numpy as np
import os

# Internal modules.
from setup.utils import get_seeds


###############################################################################


# Experiment parameter settings.

num_trials = 5
base_seed = 17264272852322
# seeds:
# Sept 26 - 22993514079364
# Oct  02 - 17264272852322
random_seeds = get_seeds(base_seed=base_seed, num=num_trials)

## Settings which are common across all runs.
paras_common = {
    "base_gpu_id": 0,
    "epochs": 500,
    "force_cpu": "no",
    "force_one_gpu": "yes",
    "loss": "CrossEntropy"
}

## Initial settings for parameters that may change depending on method.
paras_mth_defaults = {
    "adaptive": "no",
    "eta": 1.0,
    "flood_level": 0.0,
    "momentum": 0.9,
    "optimizer": "SGD",
    "optimizer_base": "none",
    "radius": 0.05,
    "sigma": 1.0,
    "step_size": 0.1,
    "theta": 0.0,
    "weight_decay": 0.0
}

## Dataset specifics.
dataset_names = ["cifar10", "cifar100", "fashionmnist", "svhn"]
num_classes_dict = {
    "cifar10": 10,
    "cifar100": 100,
    "fashionmnist": 10,
    "svhn": 10
}

## Model specifics (depends on the dataset).
model_dict = {
    "cifar10": "ResNet34",
    "cifar100": "ResNet34",
    "fashionmnist": "IshidaMLP_benchmark",
    "svhn": "ResNet18"
}

## Batch size specifics (depends on dataset).
bs_tr_dict = {
    "cifar10": 200,
    "cifar100": 200,
    "fashionmnist": 200,
    "svhn": 200
}

## Input dimensions (depends on dataset; not all models use this).
dimension_dict = {
    "cifar10": 3*32*32,
    "cifar100": 3*32*32,
    "fashionmnist": 28*28,
    "svhn": 3*32*32
}


## Methods to be evaluated.
methods = ["ERM", "Ishida", "SAM", "SoftAD"]

### Vanilla ERM.
mth_ERM = {}
mth_ERM_list = [mth_ERM]

### Ishida et al. flooding method.
mth_Ishida = {}
flood_levels = np.linspace(0.01, 1.0, 10)
mth_Ishida_list = []
for i in range(len(flood_levels)):
    to_add = copy.deepcopy(mth_Ishida)
    to_add["flood_level"] = flood_levels[i]
    mth_Ishida_list += [to_add]

### Sharpness-aware minimization (SAM).
mth_SAM = {
    "optimizer": "SAM",
    "optimizer_base": "SGD",
}
radius_values = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
mth_SAM_list = []
for i in range(len(radius_values)):
    to_add = copy.deepcopy(mth_SAM)
    to_add["radius"] = radius_values[i]
    mth_SAM_list += [to_add]

### Our SoftAD method.
mth_SoftAD = {
    "sigma": 1.0,
    "eta": 1.0
}
thetas = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75])
mth_SoftAD_list = []
for i in range(len(thetas)):
    to_add = copy.deepcopy(mth_SoftAD)
    to_add["theta"] = thetas[i]
    mth_SoftAD_list += [to_add]

mth_paras_lists = {
    "ERM": mth_ERM_list,
    "Ishida": mth_Ishida_list,
    "SAM": mth_SAM_list,
    "SoftAD": mth_SoftAD_list
}


## MLflow clerical matters.
project_uri = os.getcwd()


# Driver function.

def main():
    
    # Loop over datasets. One mlflow experiment per dataset.
    for dataset_name in dataset_names:
        
        exp_name = "exp:{}".format(dataset_name)
        exp_id = mlflow.create_experiment(exp_name)
        paras_common["dataset"] = dataset_name
        paras_common["num_classes"] = num_classes_dict[dataset_name]
        paras_common["model"] = model_dict[dataset_name]
        paras_common["bs_tr"] = bs_tr_dict[dataset_name]
        paras_common["dimension"] = dimension_dict[dataset_name]

        # Parent run for consistency with other exps (just one parent).
        with mlflow.start_run(
                run_name="parent:noise-free",
                experiment_id=exp_id
        ):
            for method in methods:
                
                paras_common["method"] = method
                mth_paras_list = mth_paras_lists[method]
                num_settings = len(mth_paras_list)
                    
                for j in range(num_settings):
                    
                    # Complete paras dict (set to defaults).
                    paras = dict(**paras_common, **paras_mth_defaults)
                    
                    # Reflect method-specific paras.
                    paras.update(mth_paras_list[j])
                    
                    # One child run for each setting (times num_trials).
                    for t in range(num_trials):
                        
                        # Make naming easy for post-processing.
                        rn = "child:{}-{}-t{}".format(method, j, t)
                        
                        # Be sure to give a fresh seed for each trial.
                        paras["random_seed"] = random_seeds[t]
                        
                        # Do the run.
                        mlflow.projects.run(uri=project_uri,
                                            entry_point="benchmarks",
                                            parameters=paras,
                                            experiment_id=exp_id,
                                            run_name=rn,
                                            env_manager="local")
    return None


if __name__ == "__main__":
    main()


###############################################################################
