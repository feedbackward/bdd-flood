'''Setup: simulated data (spiral data).'''

# External modules.
import numpy as np


###############################################################################


class Spiral:

    '''
    Generate labelled data based on a spiral labelling function.
    '''
    def __init__(self, rg, sizes, noise):
        '''
        - rg : a Numpy random generator.
        - sizes: a list of two positive integers.
        - noise: a real value between 0 and 1.
        '''
        
        # Store sample sizes.
        if len(sizes) != 2:
            raise ValueError("Tuple length not correct.")
        else:
            self.sizes = sizes
        
        # Make sure the noise make sense.
        if noise < 0.0 or noise > 1.0:
            raise ValueError("Inappropriate noise value given.")
        else:
            self.noise = noise
        
        # With checks and registration complete, store random generator.
        self.rg = rg
        
        return None
    
    
    def __call__(self):
        
        # Get the uniformly spaced radian values.
        n_pos = self.sizes[0]
        n_neg = self.sizes[1]
        r_pos = np.linspace(0.0, 4*np.pi, num=n_pos).reshape((n_pos,1))
        r_neg = np.linspace(0.0, 4*np.pi, num=n_neg).reshape((n_neg,1))
        
        # Get Gaussian noise.
        noise_pos = self.rg.multivariate_normal(mean=np.zeros(shape=2),
                                                cov=np.eye(N=2),
                                                size=self.sizes[0])
        noise_neg = self.rg.multivariate_normal(mean=np.zeros(shape=2),
                                                cov=np.eye(N=2),
                                                size=self.sizes[1])
        tau = 1.0 # default "noise level" from Ishida et al. demos.
        
        # Compute the noisy features.
        X_pos = r_pos*np.hstack([np.cos(r_pos),np.sin(r_pos)])
        X_pos += tau*noise_pos
        X_neg = (r_neg+np.pi)*np.hstack([np.cos(r_neg),np.sin(r_neg)])
        X_neg += tau*noise_neg
        X = np.vstack([X_pos,X_neg]).astype(np.float32)
        
        # Assign labels.
        Y = np.concatenate([np.full(shape=(self.sizes[0],), fill_value=1),
                            np.full(shape=(self.sizes[1],), fill_value=0)])
        Y = Y.astype(np.uint8)
        
        # If desired, randomly flip some labels.
        num_noisy = int(len(Y)*self.noise)
        if num_noisy > 0:
            noisy_idx = self.rg.choice(len(Y), size=num_noisy, replace=False)
            Y[noisy_idx] = (Y[noisy_idx] + 1) % 2
        
        # Return the full dataset as a tuple.
        return (X,Y)


###############################################################################
