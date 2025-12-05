import numpy as np
from sklearn.cluster import KMeans
import ot  # POT library for optimal transport

class DataGrouper:
    def __init__(self, config):
        self.config = config
        self.num_groups = config.num_groups
        
    def create_groups_ot(self, interaction_matrix):
        """Create data groups using Optimal Transport-based clustering"""
        num_users, num_items = interaction_matrix.shape
        
        # Compute user embeddings using SVD
        U, s, Vt = np.linalg.svd(interaction_matrix, full_matrices=False)
        k = min(20, len(s))  # Use top-k singular values
        user_embeddings = U[:, :k] @ np.diag(s[:k])
        
        # Use K-means clustering on embeddings
        kmeans = KMeans(n_clusters=self.num_groups, random_state=self.config.seed)
        group_labels = kmeans.fit_predict(user_embeddings)
        
        # Create group assignments
        groups = []
        for g in range(self.num_groups):
            user_indices = np.where(group_labels == g)[0]
            groups.append(user_indices)
            
        return groups
    
    def create_groups_random(self, num_users):
        """Create random data groups"""
        indices = np.arange(num_users)
        np.random.shuffle(indices)
        groups = np.array_split(indices, self.num_groups)
        return groups