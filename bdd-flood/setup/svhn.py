'''Setup: Street View House Numbers (SVHN) dataset.'''

# External modules.
import numpy as np
import torchvision

# Internal modules.
from setup.directories import data_path
from setup.utils import do_normalization


###############################################################################


class SVHN:
    '''
    Prepare data from SVHN dataset.
    '''

    # Label dictionary.
    label_dict = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9"
    }
    
    def __init__(self, rg, tr_frac=0.8, label_dict=label_dict,
                 data_path=data_path, download=True):
        '''
        - rg : a Numpy random generator.
        '''
        
        print("--Preparing benchmark data (SVHN)--")
        
        # Store the label dictionary just in case.
        self.label_dict = label_dict

        # Hang on to the generator.
        self.rg = rg

        # Prepare the raw data (download if needed/desired).
        data_raw_tr = torchvision.datasets.SVHN(root=data_path,
                                                split="train",
                                                download=download,
                                                transform=None)
        data_raw_te = torchvision.datasets.SVHN(root=data_path,
                                                split="test",
                                                download=download,
                                                transform=None)
        
        # Set the number of points to be used for training.
        n = len(data_raw_tr)
        self.n_tr = int(tr_frac*n)
        
        # Original index for the tr/va data.
        self.idx = np.arange(n)
        
        # Extract raw data into a more convenient form.
        self.X = np.copy(data_raw_tr.data.astype(np.float32))
        self.Y = np.copy(data_raw_tr.labels.astype(np.uint8))
        self.X_te = np.copy(data_raw_te.data.astype(np.float32))
        self.Y_te = np.copy(data_raw_te.labels.astype(np.uint8))
        del data_raw_tr, data_raw_te

        # Normalize test inputs.
        self.X_te = do_normalization(X=self.X_te)
        
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

        # Normalize the tr/va inputs.
        X_tr = np.copy(do_normalization(X=X_tr))
        X_va = np.copy(do_normalization(X=X_va))
        
        return (X_tr, Y_tr, X_va, Y_va, self.X_te, self.Y_te)
        

###############################################################################
