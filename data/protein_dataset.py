import torch
from torch.utils.data import Dataset
import os
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from tools.subgraph_sampler import ProteinSubgraphSampler
import random
import threading
from collections import OrderedDict

class ProteinDataset(Dataset):
    def __init__(self, train=True, data_path=None, transform=None, **kwargs):
        self.train = train
        self.data_path = data_path
        self.transform = transform
        
        # Subgraph sampling parameters
        self.min_nodes = kwargs.get('min_nodes', 10)
        self.max_nodes = kwargs.get('max_nodes', 100)

        # Initialize the subgraph sampler
        self.subgraph_sampler = ProteinSubgraphSampler(
            min_nodes=self.min_nodes,
            max_nodes=self.max_nodes
        )

        # Index the protein dataset
        self.file_paths = self._get_file_paths()
        self.protein_ids = self._generate_protein_ids()
        
        # Split the dataset
        self.seed = kwargs.get('seed', 42)
        self._create_data_splits()

        # Improved memory management for multiprocessing
        self.cache_size = kwargs.get('cache_size', 512)  # Reduced default for multiprocessing
        self.enable_cache = kwargs.get('enable_cache', True)
        self.preload_cache = kwargs.get('preload_cache', False)
        
        # Thread-safe cache using OrderedDict (LRU)
        self._cache_lock = threading.Lock()
        self._cache = OrderedDict()
        
        # Worker-specific settings
        self._worker_id = None
        self._is_main_process = True

    def _get_file_paths(self):
        """Get all protein graph file paths without loading them"""
        file_paths = []
            
        if self.data_path and os.path.exists(self.data_path):
            # Support multiple file formats
            pattern = '*.pt'
            file_paths.extend(glob.glob(os.path.join(self.data_path, pattern)))
            
        # Sort for consistent ordering
        file_paths.sort()
        return file_paths
    
    def _generate_protein_ids(self):
        """Generate protein IDs from file names"""
        protein_ids = []
        for file_path in self.file_paths:
            filename = Path(file_path).stem  # Get filename without extension
            protein_ids.append(filename)
        return protein_ids
    
    def _create_data_splits(self):
        """Create train/test splits and select files based on self.train"""
        if not hasattr(self, 'file_paths') or not self.file_paths:
            raise ValueError("File paths are not initialized. Please check data_path.")
        
        # Split the dataset into train and test sets
        train_files, test_files = train_test_split(
            self.file_paths, test_size=0.1, random_state=self.seed)
        
        if self.train:
            self.file_paths = train_files
        else:
            self.file_paths = test_files
        
        # Regenerate protein IDs after split
        self.protein_ids = self._generate_protein_ids()

    def _get_worker_info(self):
        """Get current worker information for multiprocessing awareness"""
        try:
            import torch.utils.data as data
            worker_info = data.get_worker_info()
            if worker_info is not None:
                self._worker_id = worker_info.id
                self._is_main_process = False
                # Adjust cache size per worker
                self.cache_size = max(32, self.cache_size // worker_info.num_workers)
            return worker_info
        except:
            return None

    def _load_single_graph(self, file_path):
        """Load a single protein graph with improved thread-safe caching"""
        if not self.enable_cache:
            return self._load_from_disk(file_path)
        
        # Get worker info on first access
        if self._worker_id is None:
            self._get_worker_info()
        
        # Thread-safe cache access
        with self._cache_lock:
            # Check cache first
            if file_path in self._cache:
                # Move to end (most recently used)
                graph_data = self._cache.pop(file_path)
                self._cache[file_path] = graph_data
                return graph_data
        
        # Load from disk if not in cache
        graph_data = self._load_from_disk(file_path)
        
        if graph_data is not None and self.enable_cache:
            self._add_to_cache(file_path, graph_data)
        
        return graph_data
    
    def _load_from_disk(self, file_path):
        """Load graph data from disk without caching"""
        try:
            graph_data = torch.load(file_path, map_location='cpu')
            
            # Validate graph data structure
            required_keys = ['node_features', 'edge_index', 'node_pos']
            if not all(key in graph_data for key in required_keys):
                raise ValueError(f"Graph data missing required keys: {required_keys}")
            
            # Ensure edge_attr exists
            if 'edge_attr' not in graph_data:
                # Create dummy edge attributes if missing
                num_edges = graph_data['edge_index'].shape[1]
                graph_data['edge_attr'] = torch.ones((num_edges, 1))
            
            return graph_data
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
        
    def _add_to_cache(self, file_path, graph_data):
        """Add graph to cache with thread-safe LRU eviction"""
        with self._cache_lock:
            # Remove oldest items if cache is full
            while len(self._cache) >= self.cache_size:
                self._cache.popitem(last=False)  # Remove oldest (FIFO)
            
            # Add new item (will be most recent)
            self._cache[file_path] = graph_data

    def clear_cache(self):
        """Clear the cache to free memory"""
        with self._cache_lock:
            self._cache.clear()

    def get_cache_info(self):
        """Get cache statistics"""
        with self._cache_lock:
            return {
                'cache_size': len(self._cache),
                'max_cache_size': self.cache_size,
                'worker_id': self._worker_id,
                'is_main_process': self._is_main_process,
                'cache_enabled': self.enable_cache
            }

    def warm_cache(self, num_samples=None):
        """Pre-load a subset of the dataset into cache"""
        if not self.enable_cache or not self._is_main_process:
            return
        
        num_samples = num_samples or min(self.cache_size, len(self.file_paths))
        print(f"Warming cache with {num_samples} samples...")
        
        # Load a random subset for better distribution
        indices = random.sample(range(len(self.file_paths)), num_samples)
        
        for i, idx in enumerate(indices):
            file_path = self.file_paths[idx]
            self._load_single_graph(file_path)
            
            if (i + 1) % 50 == 0:
                print(f"Cache warming progress: {i + 1}/{num_samples}")
        
        print(f"Cache warmed. Current cache info: {self.get_cache_info()}")

    def __len__(self):
        # Return dataset size
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Load protein graph on-demand and generate two contrastive views"""
        file_path = self.file_paths[idx]
        protein_id = self.protein_ids[idx]
            
        # Load graph data
        protein_graph = self._load_single_graph(file_path)
            
        if protein_graph is None:
            # Return dummy data if loading failed
            return {
                'protein_graph': None,
                'protein_id': protein_id,
                'idx': idx,
                'file_path': file_path,
                'load_error': True,
                'view1': None,
                'view2': None
            }
        
        # Generate two contrastive views with new sampling probabilities
        a = getattr(self, 'complete_graph_percent', 0)  # Set this attribute when initializing if needed
        complete_prob = a / 100.0
        seq_prob = (50 - 0.5 * a) / 100.0
        dist_prob = seq_prob  # Both are equal

        rand_val = random.random()
        if rand_val < complete_prob:
            # Use complete graph for both views
            view1 = self.subgraph_sampler.apply_noise(protein_graph, "identity")
            view2 = self.subgraph_sampler.apply_noise(protein_graph, "identity")
            sampling_strategy = 'complete'
        elif rand_val < complete_prob + seq_prob:
            # Sequential sampling for both views
            view1 = self.subgraph_sampler.sample_sequential_subgraph(protein_graph)
            view2 = self.subgraph_sampler.sample_sequential_subgraph(protein_graph)
            sampling_strategy = 'sequential'
        else:
            # Distance-based sampling for both views
            view1 = self.subgraph_sampler.sample_distance_subgraph(protein_graph)
            view2 = self.subgraph_sampler.sample_distance_subgraph(protein_graph)
            sampling_strategy = 'distance'

        # Randomly choose whether to apply masking (50/50 chance)
        apply_masking = random.random() < 0.5

        # Apply same noise transformation to both views
        if apply_masking:
            view1 = self.subgraph_sampler.apply_noise(view1, "random_edge_masking", edge_mask_prob=0.15)
            view2 = self.subgraph_sampler.apply_noise(view2, "random_edge_masking", edge_mask_prob=0.15)
            noise_applied = 'masking'
        else:
            view1 = self.subgraph_sampler.apply_noise(view1, "identity")
            view2 = self.subgraph_sampler.apply_noise(view2, "identity")
            noise_applied = 'identity'
                
        return {
            'protein_graph': protein_graph,
            'protein_id': protein_id,
            'idx': idx,
            'file_path': file_path,
            'load_error': False,
            'view1': view1,
            'view2': view2,
            'sampling_strategy': sampling_strategy,
            'noise_applied': noise_applied
        }