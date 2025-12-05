import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.data_path = config.data_path
        
    def load_data(self):
        """Load MovieLens 1M dataset"""
        # Read ratings file
        ratings = pd.read_csv(
            f'{self.data_path}ratings.dat',
            sep='::',
            header=None,
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            engine='python'
        )
        
        # Convert to implicit feedback (rating > 3.5 = positive)
        ratings['implicit'] = (ratings['rating'] >= 3.5).astype(int)
        
        # Create user-item interaction matrix
        users = ratings['user_id'].unique()
        items = ratings['item_id'].unique()
        
        # Map to continuous indices
        user_map = {u: i for i, u in enumerate(users)}
        item_map = {i: j for j, i in enumerate(items)}
        
        ratings['user_idx'] = ratings['user_id'].map(user_map)
        ratings['item_idx'] = ratings['item_id'].map(item_map)
        
        return ratings, len(users), len(items)
    
    def split_data(self, ratings, test_size=0.2):
        """Split data into train and test sets"""
        train_data, test_data = train_test_split(
            ratings, 
            test_size=test_size, 
            random_state=self.config.seed
        )
        return train_data, test_data
    
    def create_interaction_matrix(self, data, num_users, num_items):
        """Create sparse interaction matrix"""
        matrix = np.zeros((num_users, num_items))
        for _, row in data.iterrows():
            matrix[int(row['user_idx']), int(row['item_idx'])] = row['implicit']
        return matrix