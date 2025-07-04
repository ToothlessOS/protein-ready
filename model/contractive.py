import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean, scatter_max
from .egnn import EGNN


class ContrastiveEGNN(nn.Module):
    """
    Contrastive learning model with two EGNN blocks for protein graph pretraining.
    Each EGNN is followed by feature extraction layers for dimension reduction.
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
        
        # First EGNN block
        self.egnn1_1 = EGNN(
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

        self.egnn1_2 = EGNN(
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
        
        # Second EGNN block
        self.egnn2_1 = EGNN(
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

        self.egnn2_2 = EGNN(
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
            # Batch processing case
            batch_size = int(batch.max().item() + 1)
            
            if self.pooling == 'mean':
                return scatter_mean(node_features, batch, dim=0, dim_size=batch_size)
            elif self.pooling == 'max':
                return scatter_max(node_features, batch, dim=0, dim_size=batch_size)[0]
            elif self.pooling == 'sum':
                return scatter_add(node_features, batch, dim=0, dim_size=batch_size)
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
        # First EGNN branch - two sequential blocks
        h1, pos1 = self.egnn1_1(
            h=node_features,
            x=node_pos,
            edges=edge_index,
            edge_attr=edge_attr
        )
        
        h1, pos1 = self.egnn1_2(
            h=h1,
            x=pos1,
            edges=edge_index,
            edge_attr=edge_attr
        )
        
        # Second EGNN branch - two sequential blocks (same input, different parameters)
        h2, pos2 = self.egnn2_1(
            h=node_features,
            x=node_pos,
            edges=edge_index,
            edge_attr=edge_attr
        )
        
        h2, pos2 = self.egnn2_2(
            h=h2,
            x=pos2,
            edges=edge_index,
            edge_attr=edge_attr
        )
        
        # Graph-level pooling
        graph_features1 = self.graph_pooling(h1, batch)
        graph_features2 = self.graph_pooling(h2, batch)
        
        # Feature extraction and dimension reduction
        logits1 = self.feature_extractor1(graph_features1)
        logits2 = self.feature_extractor2(graph_features2)
        
        return logits1, logits2
    
    def get_embeddings(self, node_features, edge_index, node_pos, edge_attr, batch=None):
        """
        Get embeddings without feature extraction (for downstream tasks).
        
        Returns:
            Tuple of (embeddings1, embeddings2) from both EGNN branches
        """
        with torch.no_grad():
            # First EGNN branch - two sequential blocks
            h1, pos1 = self.egnn1_1(
                h=node_features,
                x=node_pos,
                edges=edge_index,
                edge_attr=edge_attr
            )
            
            h1, _ = self.egnn1_2(
                h=h1,
                x=pos1,
                edges=edge_index,
                edge_attr=edge_attr
            )
            
            # Second EGNN branch - two sequential blocks
            h2, pos2 = self.egnn2_1(
                h=node_features,
                x=node_pos,
                edges=edge_index,
                edge_attr=edge_attr
            )
            
            h2, _ = self.egnn2_2(
                h=h2,
                x=pos2,
                edges=edge_index,
                edge_attr=edge_attr
            )
            
            # Graph-level pooling
            embeddings1 = self.graph_pooling(h1, batch)
            embeddings2 = self.graph_pooling(h2, batch)
            
        return embeddings1, embeddings2


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for self-supervised learning.
    """
    
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2):
        """
        Compute contrastive loss between two views.
        
        Args:
            z1, z2: Embeddings from two views [batch_size, embedding_dim]
        
        Returns:
            Contrastive loss
        """
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(z1, z2.T) / self.temperature
        
        # Labels: positive pairs are on the diagonal
        batch_size = z1.size(0)
        labels = torch.arange(batch_size).to(z1.device)
        
        # Contrastive loss (InfoNCE)
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss