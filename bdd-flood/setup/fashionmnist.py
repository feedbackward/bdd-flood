'''Setup: FashionMNIST dataset.'''

# External modules.
import numpy as np
import torch
import torchvision

# Internal modules.
from setup.directories import data_path


###############################################################################


class FashionMNIST:
    '''
    Prepare data from FashionMNIST dataset.
    '''

    # Label dictionary.
    label_dict = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot"
    }
    
    def __init__(self, rg, tr_frac=0.8, flatten=True,
                 label_dict=label_dict,
                 data_path=data_path, download=True):
        '''
        - rg : a Numpy random generator.
        '''

        print("--Preparing benchmark data (FashionMNIST)--")
        
        # Store the label dictionary just in case.
        self.label_dict = label_dict

        # Hang on to the generator.
        self.rg = rg

        # Prepare the raw data (download if needed/desired).
        data_raw_tr = torchvision.datasets.FashionMNIST(root=data_path,
                                                        train=True,
                                                        download=download,
                                                        transform=None)
        data_raw_te = torchvision.datasets.FashionMNIST(root=data_path,
                                                        train=False,
                                                        download=False,
                                                        transform=None)
        
        # Set the number of points to be used for training.
        n = len(data_raw_tr)
        self.n_tr = int(tr_frac*n)
        
        # Original index for the tr/va data.
        self.idx = np.arange(n)
        
        # Extract raw data into a more convenient form.
        self.X = np.copy(data_raw_tr.data.numpy().astype(np.float32))
        self.Y = np.copy(data_raw_tr.targets.numpy().astype(np.uint8))
        self.X_te = np.copy(data_raw_te.data.numpy().astype(np.float32))
        self.Y_te = np.copy(data_raw_te.targets.numpy().astype(np.uint8))
        del data_raw_tr, data_raw_te

        # Drop the 2D structure unless specified otherwise.
        if flatten:
            height, width = self.X.shape[1:]
            self.X = np.squeeze(self.X)
            self.X = self.X.reshape(len(self.X), height*width)
            height, width = self.X_te.shape[1:]
            self.X_te = np.squeeze(self.X_te)
            self.X_te = self.X_te.reshape(len(self.X_te), height*width)
        
        return None
        
    
    def __call__(self):
        '''
        Each call gives us a chance to shuffle up the tr/va data.
        '''
        
        # Shuffle the tr/va data.
        self.rg.shuffle(self.idx)
        self.X = self.X[self.idx]
        self.Y = self.Y[self.idx]
        
        # Do the tr-va split.
        X_tr = self.X[0:self.n_tr]
        Y_tr = self.Y[0:self.n_tr]
        X_va = self.X[self.n_tr:]
        Y_va = self.Y[self.n_tr:]
        
        return (X_tr, Y_tr, X_va, Y_va, self.X_te, self.Y_te)
        

###############################################################################
