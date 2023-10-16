'''Setup: CIFAR-100 dataset.'''

# External modules.
import numpy as np
import torchvision

# Internal modules.
from setup.directories import data_path
from setup.utils import do_normalization


###############################################################################


class CIFAR100:
    '''
    Prepare data from CIFAR-100 dataset.
    '''

    # Label dictionary (fine-grained).
    label_dict = {
        0: "apple",
        1: "aquarium_fish",
        2: "baby",
        3: "bear",
        4: "beaver",
        5: "bed",
        6: "bee",
        7: "beetle",
        8: "bicycle",
        9: "bottle",
        10: "bowl",
        11: "boy",
        12: "bridge",
        13: "bus",
        14: "butterfly",
        15: "camel",
        16: "can",
        17: "castle",
        18: "caterpillar",
        19: "cattle",
        20: "chair",
        21: "chimpanzee",
        22: "clock",
        23: "cloud",
        24: "cockroach",
        25: "couch",
        26: "cra",
        27: "crocodile",
        28: "cup",
        29: "dinosaur",
        30: "dolphin",
        31: "elephant",
        32: "flatfish",
        33: "forest",
        34: "fox",
        35: "girl",
        36: "hamster",
        37: "house",
        38: "kangaroo",
        39: "keyboard",
        40: "lamp",
        41: "lawn_mower",
        42: "leopard",
        43: "lion",
        44: "lizard",
        45: "lobster",
        46: "man",
        47: "maple_tree",
        48: "motorcycle",
        49: "mountain",
        50: "mouse",
        51: "mushroom",
        52: "oak_tree",
        53: "orange",
        54: "orchid",
        55: "otter",
        56: "palm_tree",
        57: "pear",
        58: "pickup_truck",
        59: "pine_tree",
        60: "plain",
        61: "plate",
        62: "poppy",
        63: "porcupine",
        64: "possum",
        65: "rabbit",
        66: "raccoon",
        67: "ray",
        68: "road",
        69: "rocket",
        70: "rose",
        71: "sea",
        72: "seal",
        73: "shark",
        74: "shrew",
        75: "skunk",
        76: "skyscraper",
        77: "snail",
        78: "snake",
        79: "spider",
        80: "squirrel",
        81: "streetcar",
        82: "sunflower",
        83: "sweet_pepper",
        84: "table",
        85: "tank",
        86: "telephone",
        87: "television",
        88: "tiger",
        89: "tractor",
        90: "train",
        91: "trout",
        92: "tulip",
        93: "turtle",
        94: "wardrobe",
        95: "whale",
        96: "willow_tree",
        97: "wolf",
        98: "woman",
        99: "worm"
    }

    def __init__(self, rg, tr_frac=0.8, label_dict=label_dict,
                 data_path=data_path, download=True):
        '''
        - rg : a Numpy random generator.
        '''

        print("--Preparing benchmark data (CIFAR-100)--")

        # Store the label dictionary just in case.
        self.label_dict = label_dict

        # Hang on to the generator.
        self.rg = rg

        # Prepare the raw data (download if needed/desired).
        data_raw_tr = torchvision.datasets.CIFAR100(root=data_path,
                                                    train=True,
                                                    download=download,
                                                    transform=None)
        data_raw_te = torchvision.datasets.CIFAR100(root=data_path,
                                                    train=False,
                                                    download=False,
                                                    transform=None)
        
        # Set the number of points to be used for training.
        n = len(data_raw_tr)
        self.n_tr = int(tr_frac*n)
        
        # Original index for the tr/va data.
        self.idx = np.arange(n)
        
        # Extract raw data into a more convenient form.
        self.X = np.copy(data_raw_tr.data.astype(np.float32))
        self.X = self.X.transpose((0,3,1,2)) # channels in 2nd slot.
        self.Y = np.copy(np.array(data_raw_tr.targets).astype(np.uint8))
        self.X_te = np.copy(data_raw_te.data.astype(np.float32))
        self.X_te = self.X_te.transpose((0,3,1,2)) # channels in 2nd slot.
        self.Y_te = np.copy(np.array(data_raw_te.targets).astype(np.uint8))
        del data_raw_tr, data_raw_te
        
        # Normalize test inputs.
        self.X_te = np.copy(do_normalization(X=self.X_te))
        
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
