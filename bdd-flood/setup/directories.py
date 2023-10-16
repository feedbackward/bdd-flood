'''Setup: key directories used by various procedures in ours exps.'''

# External modules.
import os
from pathlib import Path


###############################################################################


# Path for storing data downloaded via torch.
data_path = os.path.join(str(Path.cwd()), "data")

# Storing images produced based on our post-learning evaluation.
img_dir_name = "img"
img_path = os.path.join(str(Path.cwd()), img_dir_name)

# Place for storing various results (aside from mlruns, if relevant).
results_dir_name = "results"
results_path = os.path.join(str(Path.cwd()), results_dir_name)


###############################################################################
