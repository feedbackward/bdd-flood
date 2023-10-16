'''Setup: simulated data (multiple Gaussians).'''

# External modules.
import numpy as np

# Internal modules.
from setup.utils import not_all_same


###############################################################################


class MultipleGaussians:
    '''
    Generate labelled data from multiple distict Gaussian distributions,
    with labels corresponding to each true underlying distribution.
    '''
    def __init__(self, rg, means, covs, sizes, noises):
        '''
        - rg : a Numpy random generator.
        - means, covs: assumed to be an iterable over
          ndarray-like objects with a "size" attribute.
        - sizes: an iterable over integers.
        - noises: an iterable over real values between 0 and 1.
        '''
        
        # Check tuple lengths.
        if not_all_same([len(means), len(covs), len(sizes), len(noises)]):
            raise ValueError("Tuple lengths don't match.")
        else:
            self.num_dists = len(means)

        # Store means and do a dimension check.
        self.means = []
        shapes = []
        for mean in means:
            self.means += [mean]
            shapes += [mean.shape]
        if not_all_same(shapes):
            raise ValueError("Mean vector dims don't match.")
        else:
            self.means = tuple(self.means)
        
        # Store covariances and do a dimension check.
        self.covs = []
        shapes = []
        for cov in covs:
            self.covs += [cov]
            shapes += [cov.shape]
        if not_all_same(shapes):
            raise ValueError("Covariance dims don't match.")
        else:
            self.covs = tuple(self.covs)
        
        # Store sample sizes.
        self.sizes = tuple(sizes)

        # Make sure the noises make sense.
        self.noises =[]
        for noise in noises:
            if noise < 0.0 or noise > 1.0:
                raise ValueError("Inappropriate noise value given.")
            else:
                self.noises += [noise]
        self.noises = tuple(self.noises)
        
        # With checks and registration complete, store random generator.
        self.rg = rg
        
        return None


    def __call__(self):
        X_list = []
        Y_list = []
        
        # Generate the data from all underlying distributions.
        for i in range(self.num_dists):
            _X = self.rg.multivariate_normal(mean=self.means[i],
                                             cov=self.covs[i],
                                             size=self.sizes[i])
            X_list += [_X.astype(np.float32)]
            y = np.full(shape=(self.sizes[i],), fill_value=i, dtype=np.uint8)
            num_noisy = int(len(y)*self.noises[i])
            if num_noisy == 0:
                Y_list += [y]
            else:
                noisy_idx = self.rg.choice(len(y), size=num_noisy,
                                           replace=False)
                noise = self.rg.integers(low=1, high=self.num_dists,
                                         size=(num_noisy,),
                                         dtype=np.uint8)
                y[noisy_idx] = (y[noisy_idx] + noise) % self.num_dists
                Y_list += [y]
        
        # Concatenate over samples.
        X = np.vstack(X_list)
        Y = np.concatenate(Y_list)
        
        # Return the full dataset as a tuple.
        return (X,Y)


###############################################################################
