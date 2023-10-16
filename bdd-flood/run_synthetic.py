'''Top script: Synthetic data tests following Ishida et al. (2020).'''

# External modules.
import copy
import mlflow
import numpy as np
import os

# Internal modules.
from setup.utils import get_seeds


###############################################################################


# Experiment parameter settings.

num_trials = 15
base_seed = 17264272852322
# seeds:
# Sept 20 - 22993514079364
# Oct  10 - 17264272852322
random_seeds = get_seeds(base_seed=base_seed, num=num_trials)

## Settings which are common across all runs.
paras_common = {
    "base_gpu_id": 0,
    "bs_tr": 50,
    "dimension": 2,
    "epochs": 500,
    "force_cpu": "no",
    "force_one_gpu": "yes",
    "loss": "CrossEntropy",
    "model": "IshidaMLP_synthetic",
    "n_tr": 100,
    "n_va": 100,
    "n_te": 20000
}

## Initial settings for parameters that may change depending on method.
paras_mth_defaults = {
    "adaptive": "no",
    "eta": 1.0,
    "flood_level": 0.0,
    "momentum": 0.0,
    "optimizer": "Adam",
    "optimizer_base": "none",
    "radius": 0.05,
    "sigma": 1.0,
    "step_size": 0.001,
    "theta": 0.0,
    "weight_decay": 0.0
}

## Dataset specifics.
dataset_names = ["gaussian", "sinusoid", "spiral"]
num_classes_dict = {
    "gaussian": 2,
    "sinusoid": 2,
    "spiral": 2
}
label_noises = [0.0, 0.05]


## Methods to be evaluated.
methods = ["ERM", "Ishida", "SAM", "SoftAD"]

### Vanilla ERM.
mth_ERM = {}
mth_ERM_list = [mth_ERM]

### Ishida et al. flooding method.
mth_Ishida = {}
flood_levels = np.linspace(0.01, 2.0, 40)
mth_Ishida_list = []
for i in range(len(flood_levels)):
    to_add = copy.deepcopy(mth_Ishida)
    to_add["flood_level"] = flood_levels[i]
    mth_Ishida_list += [to_add]

### Sharpness-aware minimization (SAM).
mth_SAM = {
    "optimizer": "SAM",
    "optimizer_base": "Adam",
}
radius_values = np.linspace(0.01, 2.0, 40)
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
thetas = np.linspace(0.01, 2.0, 40)
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
        
        for i, label_noise in enumerate(label_noises):
            
            paras_common["label_noise"] = label_noise
            
            # One parent run for each noise level.
            with mlflow.start_run(
                    run_name="parent:noise-{}".format(i),
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
                                                entry_point="synthetic",
                                                parameters=paras,
                                                experiment_id=exp_id,
                                                run_name=rn,
                                                env_manager="local")
    return None


if __name__ == "__main__":
    main()


###############################################################################
