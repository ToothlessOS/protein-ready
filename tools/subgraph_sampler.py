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