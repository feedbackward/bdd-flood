'''Setup: NSL-KDD dataset.'''

# External modules.
import numpy as np
import os
from pathlib import Path


###############################################################################


class NSL_KDD:

    '''
    The NSL-KDD dataset.
    '''

    # Hard-code path for storing downloaded and processed data.
    data_path = os.path.join(str(Path.cwd()), "data",
                             "NSL-KDD_processed", "3_minified")
    file_name = "NSL-KDD_minified.npz"
    
    
    def __init__(self, rg, noise,
                 file_path=os.path.join(data_path, file_name)):
        '''
        Really simple constructor.
        '''
        self.rg = rg
        self.file_path = file_path

        # Make sure the noise make sense.
        if noise < 0.0 or noise > 1.0:
            raise ValueError("Inappropriate noise value given.")
        else:
            self.noise = noise
        
        return None
    
    
    def __call__(self):
        
        # Read the data on file, and split into X and Y.
        with np.load(self.file_path) as data:
            Z = np.copy(data["kdd"])
            self.rg.shuffle(Z) # randomly shuffle using given rg.
            X = Z[:,0:-1]
            X = X.astype(np.float32)
            Y = Z[:,-1]
            Y = Y.astype(np.uint8)
        
        # If desired, randomly flip some labels.
        num_noisy = int(len(Y)*self.noise)
        if num_noisy > 0:
            noisy_idx = self.rg.choice(len(Y), size=num_noisy, replace=False)
            Y[noisy_idx] = (Y[noisy_idx] + 1) % 2
        
        # Return the full dataset as a tuple.
        return (X,Y)


###############################################################################
