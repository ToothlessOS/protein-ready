# Sample subgraphs from a protein graph
# which are "views" required for contrastive learning.

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import random
import os

class ProteinSubgraphSampler:
    """
    Sample subgraphs from protein graphs with sequential and distance-based strategies.
    """
    
    def __init__(self, min_nodes: int = 10, max_nodes: int = 100):
        """
        Initialize the sampler.
        
        Args:
            min_nodes: Minimum number of nodes in a subgraph
            max_nodes: Maximum number of nodes in a subgraph
        """
        if min_nodes <= 0 or max_nodes <= 0 or min_nodes > max_nodes:
            raise ValueError("Invalid node limits: min_nodes and max_nodes must be positive, min_nodes <= max_nodes")
        
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
    
    def sample_sequential_subgraph(
        self, 
        graph_data: Dict[str, torch.Tensor], 
        start_idx: Optional[int] = None,
        length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a sequential subgraph with contiguous residues.
        
        Args:
            graph_data: Original graph data with keys ['node_features', 'edge_index', 'edge_attr', 'node_pos']
            start_idx: Starting node index (random if None)
            length: Length of sequence to sample (random if None)
        
        Returns:
            Subgraph data dictionary
        """
        num_nodes = graph_data['node_features'].shape[0]
        
        # Determine sampling parameters
        if length is None:
            length = random.randint(self.min_nodes, min(self.max_nodes, num_nodes))
        
        if start_idx is None:
            max_start = max(0, num_nodes - length)
            start_idx = random.randint(0, max_start)
        
        # Ensure we don't exceed bounds
        end_idx = min(start_idx + length, num_nodes)
        actual_length = end_idx - start_idx
        
        if actual_length < self.min_nodes:
            # Adjust if too small
            start_idx = max(0, num_nodes - self.min_nodes)
            end_idx = min(start_idx + self.min_nodes, num_nodes)
        
        # Extract sequential nodes
        selected_nodes = list(range(start_idx, end_idx))
        
        return self._extract_subgraph(graph_data, selected_nodes)
    
    def sample_distance_subgraph(
        self,
        graph_data: Dict[str, torch.Tensor],
        center_idx: Optional[int] = None,
        distance_threshold: float = 10.0,
        max_nodes_override: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a distance-based subgraph around a center node.
        
        Args:
            graph_data: Original graph data
            center_idx: Center node index (random if None)
            distance_threshold: Maximum distance from center node
            max_nodes_override: Override max_nodes limit for this sampling
            
        Returns:
            Subgraph data dictionary
        """
        num_nodes = graph_data['node_features'].shape[0]
        node_pos = graph_data['node_pos']
        
        # Select center node
        if center_idx is None:
            center_idx = random.randint(0, num_nodes - 1)
        
        center_pos = node_pos[center_idx]
        
        # Calculate distances to all nodes
        distances = torch.norm(node_pos - center_pos.unsqueeze(0), dim=1)
        
        # Find nodes within distance threshold
        within_distance = torch.where(distances <= distance_threshold)[0]
        selected_nodes = within_distance.tolist()
        
        # Apply node limit
        max_limit = max_nodes_override if max_nodes_override else self.max_nodes
        if len(selected_nodes) > max_limit:
            # Keep center node and randomly sample others
            other_nodes = [n for n in selected_nodes if n != center_idx]
            random.shuffle(other_nodes)
            selected_nodes = [center_idx] + other_nodes[:max_limit-1]
        
        # Ensure minimum nodes
        if len(selected_nodes) < self.min_nodes:
            # Add closest nodes to reach minimum
            all_distances = [(i, dist.item()) for i, dist in enumerate(distances)]
            all_distances.sort(key=lambda x: x[1])
            
            for node_idx, _ in all_distances:
                if node_idx not in selected_nodes:
                    selected_nodes.append(node_idx)
                    if len(selected_nodes) >= self.min_nodes:
                        break
        
        return self._extract_subgraph(graph_data, selected_nodes)
    
    def _extract_subgraph(
        self, 
        graph_data: Dict[str, torch.Tensor], 
        selected_nodes: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract subgraph given selected node indices.
        
        Args:
            graph_data: Original graph data
            selected_nodes: List of node indices to include
            
        Returns:
            Subgraph data dictionary
        """
        if not selected_nodes:
            raise ValueError("Cannot extract subgraph: no nodes selected")
        
        # Create node mapping for efficient lookups
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_nodes)}
        selected_set = set(selected_nodes)
        
        # Extract node features and positions
        node_indices = torch.tensor(selected_nodes, dtype=torch.long)
        sub_node_features = graph_data['node_features'][node_indices]
        sub_node_pos = graph_data['node_pos'][node_indices]
        
        # Extract edges using vectorized operations
        edge_index = graph_data['edge_index']
        edge_attr = graph_data['edge_attr']
        
        # More efficient edge filtering using isin
        if edge_index.numel() > 0:
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]
            
            # Create masks for source and destination nodes
            src_mask = torch.isin(src_nodes, node_indices)
            dst_mask = torch.isin(dst_nodes, node_indices)
            edge_mask = src_mask & dst_mask
            
            if edge_mask.sum() > 0:
                sub_edge_index = edge_index[:, edge_mask]
                sub_edge_attr = edge_attr[edge_mask]
                
                # Remap edge indices efficiently
                old_to_new = torch.full((graph_data['node_features'].shape[0],), -1, dtype=torch.long)
                old_to_new[node_indices] = torch.arange(len(selected_nodes))
                
                sub_edge_index = old_to_new[sub_edge_index]
            else:
                sub_edge_index = torch.empty((2, 0), dtype=torch.long)
                sub_edge_attr = torch.empty((0, edge_attr.shape[1]), dtype=edge_attr.dtype)
        else:
            sub_edge_index = torch.empty((2, 0), dtype=torch.long)
            sub_edge_attr = torch.empty((0, edge_attr.shape[1]), dtype=edge_attr.dtype)
        
        return {
            'node_features': sub_node_features,
            'edge_index': sub_edge_index,
            'edge_attr': sub_edge_attr,
            'node_pos': sub_node_pos,
            'num_nodes': len(selected_nodes),
            'original_indices': node_indices,
            'center_node': 0 if selected_nodes else None
        }
    
    def apply_noise(
        self,
        graph_data: Dict[str, torch.Tensor],
        noise_type: str = "identity",
        edge_mask_prob: float = 0.15
    ) -> Dict[str, torch.Tensor]:
        """
        Apply noise transformation to a graph.
        
        Args:
            graph_data: Graph data dictionary
            noise_type: Type of noise to apply ("identity" or "random_edge_masking")
            edge_mask_prob: Probability of masking each edge (for random_edge_masking)
            
        Returns:
            Transformed graph data dictionary
        """
        if noise_type == "identity":
            return self._clone_graph_data(graph_data)
        elif noise_type == "random_edge_masking":
            return self._apply_edge_masking(graph_data, edge_mask_prob)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}. Supported: 'identity', 'random_edge_masking'")
    
    def _clone_graph_data(self, graph_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Clone graph data tensors."""
        return {
            'node_features': graph_data['node_features'].clone(),
            'edge_index': graph_data['edge_index'].clone(),
            'edge_attr': graph_data['edge_attr'].clone(),
            'node_pos': graph_data['node_pos'].clone(),
            'num_nodes': graph_data.get('num_nodes', graph_data['node_features'].shape[0]),
            **{k: v for k, v in graph_data.items() 
               if k not in ['node_features', 'edge_index', 'edge_attr', 'node_pos', 'num_nodes']}
        }
    
    def _apply_edge_masking(self, graph_data: Dict[str, torch.Tensor], mask_prob: float) -> Dict[str, torch.Tensor]:
        """Apply random edge masking."""
        noisy_graph = {
            'node_features': graph_data['node_features'].clone(),
            'node_pos': graph_data['node_pos'].clone(),
            'num_nodes': graph_data.get('num_nodes', graph_data['node_features'].shape[0]),
            **{k: v for k, v in graph_data.items() 
               if k not in ['node_features', 'edge_index', 'edge_attr', 'node_pos', 'num_nodes']}
        }
        
        edge_index = graph_data['edge_index']
        edge_attr = graph_data['edge_attr']
        
        if edge_index.shape[1] > 0:
            edge_mask = torch.rand(edge_index.shape[1]) > mask_prob
            noisy_graph['edge_index'] = edge_index[:, edge_mask]
            noisy_graph['edge_attr'] = edge_attr[edge_mask]
        else:
            noisy_graph['edge_index'] = edge_index.clone()
            noisy_graph['edge_attr'] = edge_attr.clone()
        
        return noisy_graph

def sample_subgraph_pair(
    sampler: ProteinSubgraphSampler,
    sampling_func,
    graph_data: Dict[str, torch.Tensor],
    sampling_type: str,
    apply_noise: bool,
    noise_type: str,
    edge_mask_prob: float,
    protein_id: str,
    subgraph_id: int,
    generate_pairs: bool,
    **kwargs
) -> List[Dict[str, torch.Tensor]]:
    """
    Sample a pair of subgraphs with the same sampling parameters for contrastive learning.
    """
    # Fix sampling parameters for consistent pair generation
    fixed_kwargs = _fix_sampling_parameters(graph_data, sampling_type, kwargs)
    
    # Generate first subgraph
    subgraph1 = sampling_func(graph_data, **fixed_kwargs)
    subgraph1.update({
        'sampling_type': sampling_type,
        'protein_id': protein_id,
        'subgraph_id': subgraph_id,
        'pair_id': 0 if generate_pairs else None
    })
    
    if apply_noise:
        subgraph1 = _apply_noise_with_metadata(sampler, subgraph1, noise_type, edge_mask_prob)
    
    if not generate_pairs:
        return [subgraph1]
    
    # Generate second subgraph with same parameters but different noise
    subgraph2 = sampling_func(graph_data, **fixed_kwargs)
    subgraph2.update({
        'sampling_type': sampling_type,
        'protein_id': protein_id,
        'subgraph_id': subgraph_id,
        'pair_id': 1
    })
    
    if apply_noise:
        subgraph2 = _apply_noise_with_metadata(sampler, subgraph2, noise_type, edge_mask_prob)
    
    return [subgraph1, subgraph2]

def _fix_sampling_parameters(graph_data: Dict[str, torch.Tensor], sampling_type: str, kwargs: dict) -> dict:
    """Fix random parameters for consistent pair generation."""
    num_nodes = graph_data['node_features'].shape[0]
    fixed_kwargs = kwargs.copy()
    
    if sampling_type == 'sequential':
        if 'start_idx' not in kwargs or kwargs['start_idx'] is None:
            length = kwargs.get('length') or random.randint(10, min(100, num_nodes))
            max_start = max(0, num_nodes - length)
            fixed_kwargs.update({
                'start_idx': random.randint(0, max_start),
                'length': length
            })
    else:  # distance sampling
        if 'center_idx' not in kwargs or kwargs['center_idx'] is None:
            fixed_kwargs['center_idx'] = random.randint(0, num_nodes - 1)
    
    return fixed_kwargs

def _apply_noise_with_metadata(
    sampler: ProteinSubgraphSampler,
    subgraph: Dict[str, torch.Tensor],
    noise_type: str,
    edge_mask_prob: float
) -> Dict[str, torch.Tensor]:
    """Apply noise to subgraph and preserve metadata."""
    # Store metadata before applying noise
    metadata = {k: v for k, v in subgraph.items() 
                if k not in ['node_features', 'edge_index', 'edge_attr', 'node_pos', 'num_nodes']}
    
    # Apply noise
    noisy_subgraph = sampler.apply_noise(subgraph, noise_type, edge_mask_prob)
    
    # Restore metadata and add noise info
    noisy_subgraph.update(metadata)
    noisy_subgraph['noise_applied'] = noise_type
    if noise_type == "random_edge_masking":
        noisy_subgraph['edge_mask_prob'] = edge_mask_prob
    
    return noisy_subgraph

def batch_subgraphs(subgraphs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Batch multiple subgraphs into a single data structure for efficient processing.
    """
    if not subgraphs:
        return {}
    
    # Pre-allocate lists for better performance
    batch_node_features = []
    batch_edge_index = []
    batch_edge_attr = []
    batch_node_pos = []
    batch_indices = []
    
    # Metadata tracking
    metadata = {
        'sampling_types': [],
        'noise_applied': [],
        'protein_ids': [],
        'subgraph_ids': [],
        'pair_ids': []
    }
    
    node_offset = 0
    
    for i, subgraph in enumerate(subgraphs):
        num_nodes = subgraph['num_nodes']
        
        # Collect node data
        batch_node_features.append(subgraph['node_features'])
        batch_node_pos.append(subgraph['node_pos'])
        batch_indices.extend([i] * num_nodes)
        
        # Handle edges with offset adjustment
        if subgraph['edge_index'].shape[1] > 0:
            adj_edge_index = subgraph['edge_index'] + node_offset
            batch_edge_index.append(adj_edge_index)
            batch_edge_attr.append(subgraph['edge_attr'])
        
        # Collect metadata
        metadata['sampling_types'].append(subgraph.get('sampling_type', 'unknown'))
        metadata['noise_applied'].append(subgraph.get('noise_applied', 'none'))
        metadata['protein_ids'].append(subgraph.get('protein_id', f'unknown_{i}'))
        metadata['subgraph_ids'].append(subgraph.get('subgraph_id', i))
        metadata['pair_ids'].append(subgraph.get('pair_id', None))
        
        node_offset += num_nodes
    
    # Concatenate tensors
    batched_data = {
        'node_features': torch.cat(batch_node_features, dim=0),
        'node_pos': torch.cat(batch_node_pos, dim=0),
        'batch': torch.tensor(batch_indices, dtype=torch.long),
        **metadata
    }
    
    # Handle edges (may be empty for some subgraphs)
    if batch_edge_index:
        batched_data['edge_index'] = torch.cat(batch_edge_index, dim=1)
        batched_data['edge_attr'] = torch.cat(batch_edge_attr, dim=0)
    else:
        # Create empty edge tensors with proper shape
        edge_attr_dim = subgraphs[0]['edge_attr'].shape[1]
        batched_data['edge_index'] = torch.empty((2, 0), dtype=torch.long)
        batched_data['edge_attr'] = torch.empty((0, edge_attr_dim), dtype=subgraphs[0]['edge_attr'].dtype)
    
    return batched_data

def sample_multiple_subgraphs(
    graph_data: Dict[str, torch.Tensor],
    sampler: ProteinSubgraphSampler,
    num_sequential: int = 5,
    num_distance: int = 5,
    apply_noise: bool = False,
    noise_type: str = "identity",
    edge_mask_prob: float = 0.15,
    return_batch: bool = True,
    generate_pairs: bool = False,
    protein_id: Optional[str] = None,
    **kwargs
) -> Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """
    Sample multiple subgraphs from a single protein graph with optimizations for batch processing.
    """
    if protein_id is None:
        protein_id = f"protein_{id(graph_data)}"
    
    subgraphs = []
    subgraph_id = 0
    
    # Sample sequential subgraphs
    for _ in range(num_sequential):
        subgraph_pair = sample_subgraph_pair(
            sampler, sampler.sample_sequential_subgraph, graph_data, 'sequential',
            apply_noise, noise_type, edge_mask_prob, protein_id, subgraph_id, generate_pairs, **kwargs
        )
        subgraphs.extend(subgraph_pair)
        subgraph_id += 1
    
    # Sample distance-based subgraphs
    for _ in range(num_distance):
        subgraph_pair = sample_subgraph_pair(
            sampler, sampler.sample_distance_subgraph, graph_data, 'distance',
            apply_noise, noise_type, edge_mask_prob, protein_id, subgraph_id, generate_pairs, **kwargs
        )
        subgraphs.extend(subgraph_pair)
        subgraph_id += 1
    
    return batch_subgraphs(subgraphs) if return_batch else subgraphs

def create_subgraph_dataloader_batch(
    protein_graphs: List[Dict[str, torch.Tensor]],
    sampler: ProteinSubgraphSampler,
    subgraphs_per_protein: int = 8,
    sequential_ratio: float = 0.5,
    apply_noise: bool = True,
    noise_types: List[str] = ["identity", "random_edge_masking"],
    edge_mask_prob: float = 0.15,
    generate_pairs: bool = True,
    protein_ids: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Create a batch of subgraphs from multiple protein graphs for efficient dataloader processing.
    """
    if not protein_graphs:
        raise ValueError("Cannot create batch from empty protein_graphs list")
    
    all_subgraphs = []
    
    # Calculate subgraph counts
    actual_samples = subgraphs_per_protein // 2 if generate_pairs else subgraphs_per_protein
    num_sequential = int(actual_samples * sequential_ratio)
    num_distance = actual_samples - num_sequential
    
    for i, protein_graph in enumerate(protein_graphs):
        protein_id = protein_ids[i] if protein_ids and i < len(protein_ids) else f"protein_{i}"
        noise_type = random.choice(noise_types) if noise_types else "identity"
        
        protein_subgraphs = sample_multiple_subgraphs(
            protein_graph, sampler,
            num_sequential=num_sequential,
            num_distance=num_distance,
            apply_noise=apply_noise,
            noise_type=noise_type,
            edge_mask_prob=edge_mask_prob,
            return_batch=False,
            generate_pairs=generate_pairs,
            protein_id=protein_id,
            **kwargs
        )
        
        all_subgraphs.extend(protein_subgraphs)
    
    return batch_subgraphs(all_subgraphs)

if __name__ == "__main__":
    # Example usage
    sampler = ProteinSubgraphSampler(min_nodes=20, max_nodes=80)
    
    # Load a protein graph
    graph_path = "../dataset/protein_g/pytorch_graph_1kar_HSM_A_502.pt"
    if os.path.exists(graph_path):
        graph_data = torch.load(graph_path)
        
        # Sample different types of subgraphs
        sequential_sub = sampler.sample_sequential_subgraph(graph_data, length=50)
        distance_sub = sampler.sample_distance_subgraph(graph_data, distance_threshold=15.0)
        
        # Apply noise transformations
        identity_sub = sampler.apply_noise(sequential_sub, "identity")
        masked_sub = sampler.apply_noise(distance_sub, "random_edge_masking", edge_mask_prob=0.15)
        
        print(f"Original graph: {graph_data['node_features'].shape[0]} nodes")
        print(f"Sequential subgraph: {sequential_sub['num_nodes']} nodes")
        print(f"Distance subgraph: {distance_sub['num_nodes']} nodes")
        print(f"Original edges in distance subgraph: {distance_sub['edge_index'].shape[1]}")
        print(f"Edges after masking: {masked_sub['edge_index'].shape[1]}")