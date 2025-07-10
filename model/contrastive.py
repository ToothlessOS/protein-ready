import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from .egnn import EGNN


class ContrastiveEGNN(nn.Module):
    """
    Contrastive learning model with one EGNN block for protein graph pretraining.
    The EGNN is followed by two different feature extraction layers for contrastive learning.
    """
    
    def __init__(self, 
                 in_node_nf=960,  # ESM embedding dimension
                 in_edge_nf=11,    # Edge feature dimension from build_protein_graph.py
                 hidden_nf=512,
                 egnn_layers=4,
                 output_dim=64,
                 projection_dim=96,
                 device='cuda',
                 pooling='mean'):
        """
        Args:
            in_node_nf: Input node feature dimension (ESM embeddings)
            in_edge_nf: Input edge feature dimension
            hidden_nf: Hidden dimension for EGNN
            egnn_layers: Number of EGNN layers
            output_dim: Final output dimension
            device: Device to run the model on
            pooling: Graph pooling method ('mean', 'max', 'sum')
        """
        super(ContrastiveEGNN, self).__init__()
        
        self.device = device
        self.pooling = pooling
        
        # Single EGNN block
        self.egnn1 = EGNN(
            in_node_nf=in_node_nf,
            hidden_nf=hidden_nf,
            out_node_nf=hidden_nf,
            in_edge_nf=in_edge_nf,
            device=device,
            n_layers=egnn_layers,
            residual=True,
            attention=True,
            normalize=True
        )

        self.egnn2 = EGNN(
            in_node_nf=hidden_nf,
            hidden_nf=hidden_nf//4,
            out_node_nf=hidden_nf//4,
            in_edge_nf=in_edge_nf,
            device=device,
            n_layers=egnn_layers,
            residual=True,
            attention=True,
            normalize=True
        )
        
        # Feature extraction layers for first branch
        self.feature_extractor1 = nn.Sequential(
            nn.Linear(hidden_nf//4, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, output_dim)
        )
        
        # Feature extraction layers for second branch
        self.feature_extractor2 = nn.Sequential(
            nn.Linear(hidden_nf//4, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, output_dim)
        )
        
        self.to(device)
    
    def graph_pooling(self, node_features, batch=None):
        """
        Pool node features to get graph-level representation.
        
        Args:
            node_features: Node features tensor [num_nodes, feature_dim]
            batch: Batch tensor indicating which graph each node belongs to [num_nodes]
        
        Returns:
            Graph-level features [batch_size, feature_dim]
        """
        if batch is None:
            # Single graph case
            if self.pooling == 'mean':
                return torch.mean(node_features, dim=0, keepdim=True)
            elif self.pooling == 'max':
                return torch.max(node_features, dim=0, keepdim=True)[0]
            elif self.pooling == 'sum':
                return torch.sum(node_features, dim=0, keepdim=True)
        else:
            # Batch processing case using PyTorch Geometric pooling functions
            if self.pooling == 'mean':
                return global_mean_pool(node_features, batch)
            elif self.pooling == 'max':
                return global_max_pool(node_features, batch)
            elif self.pooling == 'sum':
                return global_add_pool(node_features, batch)
            else:
                raise ValueError(f"Unsupported pooling method: {self.pooling}")
    
    def forward(self, node_features, edge_index, node_pos, edge_attr, batch=None):
        """
        Forward pass through the contrastive model.
        
        Args:
            node_features: Node feature matrix [num_nodes, in_node_nf]
            edge_index: Edge indices [2, num_edges]
            node_pos: Node coordinates [num_nodes, 3]
            edge_attr: Edge attributes [num_edges, in_edge_nf]
            batch: Batch tensor for multiple graphs [num_nodes]
        
        Returns:
            Tuple of (logits1, logits2) for contrastive learning
        """
        # Single EGNN processing - two sequential blocks
        h, pos = self.egnn1(
            h=node_features,
            x=node_pos,
            edges=edge_index,
            edge_attr=edge_attr
        )
        
        h, pos = self.egnn2(
            h=h,
            x=pos,
            edges=edge_index,
            edge_attr=edge_attr
        )
        
        # Graph-level pooling
        graph_features = self.graph_pooling(h, batch)
        
        # Feature extraction and dimension reduction using two different extractors
        logits1 = self.feature_extractor1(graph_features)
        logits2 = self.feature_extractor2(graph_features)
        
        return logits1, logits2
    
    def get_embeddings(self, node_features, edge_index, node_pos, edge_attr, batch=None):
        """
        Get embeddings without feature extraction (for downstream tasks).
        
        Returns:
            Tuple of (embeddings1, embeddings2) from both feature extractors
        """
        with torch.no_grad():
            # Single EGNN processing - two sequential blocks
            h, pos = self.egnn1(
                h=node_features,
                x=node_pos,
                edges=edge_index,
                edge_attr=edge_attr
            )
            
            h, _ = self.egnn2(
                h=h,
                x=pos,
                edges=edge_index,
                edge_attr=edge_attr
            )
            
            # Graph-level pooling
            embeddings = self.graph_pooling(h, batch)
            
        return embeddings, embeddings


class ContrastiveLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss as proposed in SimCLR.
    """
    
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2):
        """
        Compute NT-Xent loss between two views.
        
        Args:
            z1, z2: Embeddings from two views [batch_size, embedding_dim]
        
        Returns:
            NT-Xent contrastive loss
        """
        batch_size = z1.size(0)
        
        # Step 1: Normalize embeddings to unit vectors
        # This ensures cosine similarity is computed correctly
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Step 2: Concatenate embeddings from both views
        # Resulting tensor shape: [2*batch_size, embedding_dim]
        z = torch.cat([z1, z2], dim=0)
        
        # Step 3: Compute pairwise cosine similarity matrix
        # Shape: [2*batch_size, 2*batch_size]
        # Divide by temperature to scale the similarities
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        # Step 4: Create a mask to remove self-similarities (diagonal entries)
        # Self-similarities are not considered in the loss calculation
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        
        # Step 5: Identify positive pairs
        # Positive pairs are (i, i+batch_size) and (i+batch_size, i)
        # These correspond to embeddings from the same sample but different augmentations
        pos_indices = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),  # for z1
            torch.arange(0, batch_size, device=z.device)  # for z2
        ])
        
        # Step 6: Extract positive similarities from the similarity matrix
        # Positive similarities are located at specific indices in the matrix
        pos_sim = sim_matrix[torch.arange(2 * batch_size, device=z.device), pos_indices]
        
        # Step 7: Compute the denominator using log-sum-exp
        # Denominator includes all similarities except self-similarities
        exp_sim = torch.exp(sim_matrix)
        sum_exp_sim = exp_sim.sum(dim=1)
        
        # Step 8: Compute NT-Xent loss for each sample
        # Loss formula: -log(exp(pos_sim) / sum(exp(all_sim_except_self)))
        loss = -torch.log(torch.exp(pos_sim) / sum_exp_sim)
        
        # Step 9: Return the mean loss across all samples
        return loss.mean()


# Compatibility alias for the naming convention used by MInterface
# The framework expects snake_case -> CamelCase conversion
# So 'contrastive' -> 'Contrastive'
Contrastive = ContrastiveEGNN