import os
import numpy as np

class Config:
    def __init__(self):
        # Dataset configuration
        self.dataset = 'ml1m'
        self.data_path = './data/ml1m/'
        
        # Model parameters
        self.num_factors = 20  # Dimensionality of latent factors
        self.lr = 0.01  # Learning rate
        self.reg = 0.01  # Regularization parameter
        self.epochs = 50
        self.batch_size = 256
        
        # Unlearning parameters
        self.num_groups = 5  # Number of data shards
        self.del_percentage = 2  # Percentage of data to unlearn
        self.del_type = 'rand'  # Type of deletion: 'rand' or 'targeted'
        
        # Training settings
        self.device = 'mps' if torch.cuda.is_available() else 'cpu'  # Use MPS for M4
        self.seed = 42
        self.verbose = 1

import torch
config = Config()