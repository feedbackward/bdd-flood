'''Setup: basic config for visualization of experimental results.'''

# External modules.
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Internal modules.
from setup.directories import img_path


###############################################################################


# Figure-related parameters.
my_ext = "pdf"


# Handy utilities.

def get_img_path(img_name: str) -> str:
    return os.path.join(img_path, img_name)


def makedir_safe(dirname: str) -> None:
    '''
    A simple utility for making new directories
    after checking that they do not exist.
    '''
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return None


def save_figure(figname: str) -> None:
    makedir_safe(img_path)
    fname = ".".join([os.path.join(img_path, figname), _my_ext])
    plt.savefig(fname=fname)
    return None


## A handy routine for saving just a legend.
def export_legend(legend, filename="legend.pdf", expand=[-5,-5,5,5]):
    '''
    Save just the legend.
    Source for this: https://stackoverflow.com/a/47749903
    '''
    fig = legend.figure
    fig.canvas.draw()
    plt.axis('off')
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)
    return None


###############################################################################
