"""
Custom collate function for graph data with variable sizes.
"""

import torch
from torch.utils.data.dataloader import default_collate
from typing import List, Dict, Any, Union


def graph_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for protein graph datasets.
    
    Handles variable-sized graphs by creating batched tensors with appropriate indexing.
    
    Args:
        batch: List of sample dictionaries from the dataset
        
    Returns:
        Batched dictionary with properly collated graph data
    """
    # Handle empty batch
    if not batch:
        return {}
    
    # Check for load errors
    valid_samples = [sample for sample in batch if not sample.get('load_error', False)]
    if not valid_samples:
        # All samples failed to load, return error batch
        return {
            'load_error': True,
            'batch_size': len(batch),
            'error_msg': 'All samples in batch failed to load'
        }
    
    # Use only valid samples
    batch = valid_samples
    batch_size = len(batch)
    
    # Initialize result dictionary
    result = {
        'batch_size': batch_size,
        'load_error': False
    }
    
    # Collate simple fields (strings, integers, etc.)
    simple_fields = ['protein_id', 'idx', 'file_path', 'sampling_strategy', 'noise_applied']
    for field in simple_fields:
        if field in batch[0]:
            result[field] = [sample[field] for sample in batch]
    
    # Collate graph views
    for view_name in ['view1', 'view2']:
        if view_name in batch[0] and batch[0][view_name] is not None:
            result[view_name] = collate_graph_data([sample[view_name] for sample in batch])
    
    # Collate original protein graph if needed
    if 'protein_graph' in batch[0] and batch[0]['protein_graph'] is not None:
        result['protein_graph'] = collate_graph_data([sample['protein_graph'] for sample in batch])
    
    return result


def collate_graph_data(graph_list: List[Dict[str, torch.Tensor]]) -> Dict[str, Union[torch.Tensor, List[int]]]:
    """
    Collate a list of graph dictionaries into a batched format.
    
    Args:
        graph_list: List of graph dictionaries, each containing:
            - node_features: [num_nodes, feature_dim]
            - edge_index: [2, num_edges] 
            - edge_attr: [num_edges, edge_feature_dim]
            - node_pos: [num_nodes, 3]
            
    Returns:
        Batched graph dictionary with:
            - node_features: [total_nodes, feature_dim]
            - edge_index: [2, total_edges]
            - edge_attr: [total_edges, edge_feature_dim] 
            - node_pos: [total_nodes, 3]
            - batch: [total_nodes] - batch assignment for each node
            - num_nodes_per_graph: [batch_size] - number of nodes in each graph
            - num_edges_per_graph: [batch_size] - number of edges in each graph
    """
    if not graph_list:
        return {}
    
    # Filter out None graphs
    valid_graphs = [g for g in graph_list if g is not None]
    if not valid_graphs:
        return {}
    
    batch_size = len(valid_graphs)
    
    # Collect node features
    node_features_list = []
    node_pos_list = []
    edge_index_list = []
    edge_attr_list = []
    
    batch_assignment = []
    num_nodes_per_graph = []
    num_edges_per_graph = []
    
    node_offset = 0
    
    for batch_idx, graph in enumerate(valid_graphs):
        # Validate required keys
        required_keys = ['node_features', 'edge_index', 'node_pos']
        if not all(key in graph for key in required_keys):
            continue
            
        num_nodes = graph['node_features'].shape[0]
        num_edges = graph['edge_index'].shape[1]
        
        # Collect node features and positions
        node_features_list.append(graph['node_features'])
        node_pos_list.append(graph['node_pos'])
        
        # Adjust edge indices to account for batching
        edge_index = graph['edge_index'] + node_offset
        edge_index_list.append(edge_index)
        
        # Collect edge attributes
        if 'edge_attr' in graph and graph['edge_attr'] is not None:
            edge_attr_list.append(graph['edge_attr'])
        else:
            # Create dummy edge attributes if missing
            edge_attr_list.append(torch.ones((num_edges, 1), dtype=torch.float32))
        
        # Create batch assignment for nodes
        batch_assignment.extend([batch_idx] * num_nodes)
        
        # Track sizes
        num_nodes_per_graph.append(num_nodes)
        num_edges_per_graph.append(num_edges)
        
        # Update offset for next graph
        node_offset += num_nodes
    
    # Concatenate all features
    result = {}
    
    if node_features_list:
        result['node_features'] = torch.cat(node_features_list, dim=0)
    
    if node_pos_list:
        result['node_pos'] = torch.cat(node_pos_list, dim=0)
    
    if edge_index_list:
        result['edge_index'] = torch.cat(edge_index_list, dim=1)
    
    if edge_attr_list:
        result['edge_attr'] = torch.cat(edge_attr_list, dim=0)
    
    # Add batch information
    result['batch'] = torch.tensor(batch_assignment, dtype=torch.long)
    result['num_nodes_per_graph'] = num_nodes_per_graph
    result['num_edges_per_graph'] = num_edges_per_graph
    result['batch_size'] = batch_size
    
    return result


def unbatch_graph_data(batched_graph: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    """
    Convert batched graph data back to individual graphs.
    
    Args:
        batched_graph: Batched graph dictionary from collate_graph_data
        
    Returns:
        List of individual graph dictionaries
    """
    if not batched_graph or 'batch' not in batched_graph:
        return []
    
    batch_size = batched_graph['batch_size']
    num_nodes_per_graph = batched_graph['num_nodes_per_graph']
    num_edges_per_graph = batched_graph['num_edges_per_graph']
    
    graphs = []
    node_offset = 0
    edge_offset = 0
    
    for i in range(batch_size):
        num_nodes = num_nodes_per_graph[i]
        num_edges = num_edges_per_graph[i]
        
        graph = {}
        
        # Extract node features and positions
        if 'node_features' in batched_graph:
            graph['node_features'] = batched_graph['node_features'][node_offset:node_offset + num_nodes]
        
        if 'node_pos' in batched_graph:
            graph['node_pos'] = batched_graph['node_pos'][node_offset:node_offset + num_nodes]
        
        # Extract and adjust edge indices
        if 'edge_index' in batched_graph:
            edge_index = batched_graph['edge_index'][:, edge_offset:edge_offset + num_edges]
            graph['edge_index'] = edge_index - node_offset
        
        # Extract edge attributes
        if 'edge_attr' in batched_graph:
            graph['edge_attr'] = batched_graph['edge_attr'][edge_offset:edge_offset + num_edges]
        
        graphs.append(graph)
        
        # Update offsets
        node_offset += num_nodes
        edge_offset += num_edges
    
    return graphs
