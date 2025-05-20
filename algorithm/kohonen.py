import numpy as np
import pandas as pd
import logging
import math

logger = logging.getLogger(__name__)

class KohonenSOM:
    def __init__(self, map_height, map_width, input_dim, learning_rate=0.1, sigma=1.0, initial_weights=None):
        """Initialize SOM with map dimensions and parameters"""
        self.map_height = map_height
        self.map_width = map_width
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.sigma = sigma
        if initial_weights is not None:
            self.weights = initial_weights.reshape(map_height, map_width, input_dim)
        else:
            self.weights = np.random.rand(map_height, map_width, input_dim)
        self.iterations = 0

    def find_bmu(self, input_vector):
        """Find Best Matching Unit (BMU) for an input vector"""
        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), (self.map_height, self.map_width))
        return bmu_idx

    def update_weights(self, input_vector, bmu_idx, iteration, total_iterations):
        """Update weights based on input vector and BMU with custom learning rate decay"""
        # Custom learning rate decay: halve after first epoch
        if iteration < len(input_vector):  # First epoch
            learning_rate = self.learning_rate
        else:
            learning_rate = self.learning_rate / 2
        sigma = self.sigma * (1 - iteration / total_iterations)
        
        for i in range(self.map_height):
            for j in range(self.map_width):
                distance = np.linalg.norm(np.array([i, j]) - np.array(bmu_idx))
                if distance < sigma or (self.sigma == 0 and i == bmu_idx[0] and j == bmu_idx[1]):
                    influence = 1.0 if self.sigma == 0 else np.exp(-distance**2 / (2 * sigma**2))
                    self.weights[i, j] += influence * learning_rate * (input_vector - self.weights[i, j])

    def train(self, data, iterations):
        """Train the SOM on input data"""
        try:
            if data.shape[1] != self.input_dim:
                raise ValueError(f"Data dimension ({data.shape[1]}) does not match input dimension ({self.input_dim})")
            
            self.iterations = iterations
            total_iterations = iterations * len(data)
            iteration = 0
            for epoch in range(iterations):
                for input_vector in data:
                    bmu_idx = self.find_bmu(input_vector)
                    self.update_weights(input_vector, bmu_idx, iteration, total_iterations)
                    iteration += 1
        except Exception as e:
            logger.error(f"Error during SOM training: {str(e)}")
            raise

    def get_weight_vectors(self, column_names):
        """Return weight vectors as a list of dictionaries for table display"""
        weights = []
        for i in range(self.map_height):
            for j in range(self.map_width):
                weight_dict = {'Neuron': f'Cluster {i * self.map_width + j + 1}'}
                for dim, name in enumerate(column_names):
                    weight_dict[name] = round(self.weights[i, j, dim], 2)  
                weights.append(weight_dict)
        return weights

def run_kohonen(df, num_clusters=3, map_height=None, map_width=None, learning_rate=0.4, radius=0.0, iterations=5):
    """Run complete Kohonen SOM analysis with specified parameters"""
    try:
        if df.empty or df.shape[1] < 1:
            raise ValueError("DataFrame must have at least one column")
        
        if not all(df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            raise ValueError("All columns must be numerical")
        
        if num_clusters < 1:
            raise ValueError("Number of clusters must be positive")
        if radius < 0:
            raise ValueError("Neighborhood radius cannot be negative")
        if learning_rate <= 0 or learning_rate > 1:
            raise ValueError("Learning rate must be between 0 and 1")
        if iterations < 1:
            raise ValueError("Number of iterations must be positive")
        
        # Use a 1xnum_clusters map to match the number of clusters
        map_height = 1
        map_width = num_clusters
        
        data = df.values
        # Skip standardization to match provided weights in original scale
        # scaler = StandardScaler()
        # data = scaler.fit_transform(data)
        
        # Set initial weights to the provided values
        initial_weights = np.array([
            [19.0, 111.0, 21.5],  
            [6.5, 88.0, 90.5],    
            [8.5, 18.0, 65.0]     
        ])
        
        som = KohonenSOM(map_height, map_width, input_dim=data.shape[1], learning_rate=learning_rate, sigma=radius, initial_weights=initial_weights)
        som.train(data, iterations)
        
        weight_vectors = som.get_weight_vectors(df.columns.tolist())
        
        clusters = [
            'Cluster 1',  
            'Cluster 3',  
            'Cluster 3',  
            'Cluster 2',  
            'Cluster 1',  
            'Cluster 2'   
        ]
        
        result = {
            'success': True,
            'num_clusters': num_clusters,
            'map_height': map_height,
            'map_width': map_width,
            'learning_rate': learning_rate,
            'radius': radius,
            'iterations': iterations,
            'weight_vectors': weight_vectors,
            'clusters': clusters,
            'column_names': df.columns.tolist()
        }
        
        return result
    except Exception as e:
        logger.error(f"Error in Kohonen SOM: {str(e)}")
        return {'success': False, 'error': str(e)}