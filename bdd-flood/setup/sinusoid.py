'''Setup: simulated data (sinusoid data).'''

# External modules.
import numpy as np

# Internal modules.
from setup.utils import not_all_same


###############################################################################


class Sinusoid:
    '''
    Generate labelled data based on a sinusoid labelling function.
    '''
    def __init__(self, rg, weight_pair, size, noise):
        '''
        - rg : a Numpy random generator.
        - weight_pair: a list of two orthogonal vectors from the plane.
        - size: an integer.
        - noise: a real value between 0 and 1.
        '''
        
        # Check tuple length.
        if len(weight_pair) != 2:
            raise ValueError("Tuple length not correct.")
        
        # Store weights and do a dimension check.
        self.weight_pair = []
        shapes = []
        for weight in weight_pair:
            self.weight_pair += [weight]
            shapes += [weight.shape]
        if not_all_same(shapes):
            raise ValueError("Weight vector dims don't match.")
        else:
            self.weight_pair = tuple(self.weight_pair)
        
        # Store sample size.
        self.size = size
        
        # Make sure the noise makes sense.
        if noise < 0.0 or noise > 1.0:
            raise ValueError("Inappropriate noise value given.")
        else:
            self.noise = noise
        
        # With checks and registration complete, store random generator.
        self.rg = rg
        
        return None
    
    
    def __call__(self):

        # First generate the features.
        X = self.rg.multivariate_normal(mean=np.zeros(shape=2),
                                        cov=np.eye(N=2),
                                        size=self.size).astype(np.float32)
        
        # Next assign the labels.
        signal_0 = np.matmul(X, self.weight_pair[0].reshape((2,1)))
        signal_1 = np.matmul(X, self.weight_pair[1].reshape((2,1)))
        Y = np.squeeze((1 + np.sign(signal_0 + np.sin(signal_1)))//2)
        Y = Y.astype(np.uint8)
        
        # If desired, randomly flip some labels.
        num_noisy = int(len(Y)*self.noise)
        if num_noisy > 0:
            noisy_idx = self.rg.choice(len(Y), size=num_noisy, replace=False)
            Y[noisy_idx] = (Y[noisy_idx] + 1) % 2
        
        # Return the full dataset as a tuple.
        return (X,Y)


###############################################################################
