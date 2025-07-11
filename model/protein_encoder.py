"""
Protein Encoder Module for Feature Extraction and Downstream Tasks

This module provides a clean interface for using the pretrained protein representation
model for downstream tasks like protein classification, property prediction, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from .egnn import EGNN
from .contrastive import ContrastiveEGNN


class ProteinEncoder(nn.Module):
    """
    Protein encoder that can be used for downstream tasks.
    Can be initialized from a pretrained contrastive model or trained from scratch.
    """
    
    def __init__(self, 
                 in_node_nf=960,  # ESM embedding dimension
                 in_edge_nf=11,   # Edge feature dimension
                 hidden_nf=512,
                 egnn_layers=4,
                 device='cuda',
                 pooling='mean',
                 freeze_backbone=False):
        """
        Args:
            in_node_nf: Input node feature dimension (ESM embeddings)
            in_edge_nf: Input edge feature dimension
            hidden_nf: Hidden dimension for EGNN
            egnn_layers: Number of EGNN layers
            device: Device to run the model on
            pooling: Graph pooling method ('mean', 'max', 'sum')
            freeze_backbone: Whether to freeze the EGNN backbone for fine-tuning
        """
        super(ProteinEncoder, self).__init__()
        
        self.device = device
        self.pooling = pooling
        self.freeze_backbone = freeze_backbone
        self.hidden_nf = hidden_nf
        
        # EGNN backbone
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
        
        # Set gradient requirements based on freeze_backbone
        if self.freeze_backbone:
            self._freeze_backbone()
        
        self.to(device)
    
    def _freeze_backbone(self):
        """Freeze the EGNN backbone parameters for fine-tuning scenarios."""
        for param in self.egnn1.parameters():
            param.requires_grad = False
        for param in self.egnn2.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone for end-to-end training."""
        self.freeze_backbone = False
        for param in self.egnn1.parameters():
            param.requires_grad = True
        for param in self.egnn2.parameters():
            param.requires_grad = True
    
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
    
    def forward(self, node_features, edge_index, node_pos, edge_attr, batch=None, return_node_features=False):
        """
        Forward pass through the encoder.
        
        Args:
            node_features: Node feature matrix [num_nodes, in_node_nf]
            edge_index: Edge indices [2, num_edges]
            node_pos: Node coordinates [num_nodes, 3]
            edge_attr: Edge attributes [num_edges, in_edge_nf]
            batch: Batch tensor for multiple graphs [num_nodes]
            return_node_features: Whether to return node-level features in addition to graph-level
        
        Returns:
            Graph-level embeddings [batch_size, hidden_nf//4]
            If return_node_features=True, also returns node-level features
        """
        # EGNN processing
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
        graph_embeddings = self.graph_pooling(h, batch)
        
        if return_node_features:
            return graph_embeddings, h, pos
        else:
            return graph_embeddings
    
    @classmethod
    def from_pretrained(cls, checkpoint_path, freeze_backbone=False, **kwargs):
        """
        Create a ProteinEncoder from a pretrained contrastive model checkpoint.
        
        Args:
            checkpoint_path: Path to the pretrained model checkpoint
            freeze_backbone: Whether to freeze the backbone for fine-tuning
            **kwargs: Additional arguments for the encoder
        
        Returns:
            ProteinEncoder instance with pretrained weights
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract model hyperparameters from checkpoint
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
            # Update kwargs with checkpoint hyperparameters
            for key in ['in_node_nf', 'in_edge_nf', 'hidden_nf', 'egnn_layers', 'pooling']:
                if key in hparams and key not in kwargs:
                    kwargs[key] = hparams[key]
        
        # Create encoder instance
        encoder = cls(freeze_backbone=freeze_backbone, **kwargs)
        
        # Load pretrained weights
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            
            # Filter state dict to only include EGNN weights
            encoder_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.egnn1.') or key.startswith('model.egnn2.'):
                    # Remove 'model.' prefix
                    new_key = key[6:]  # Remove 'model.'
                    encoder_state_dict[new_key] = value
            
            # Load the filtered state dict
            encoder.load_state_dict(encoder_state_dict, strict=False)
            print(f"Loaded pretrained weights from {checkpoint_path}")
            print(f"Loaded {len(encoder_state_dict)} parameters")
        
        return encoder


class ProteinClassifier(nn.Module):
    """
    Protein classifier for downstream classification tasks.
    """
    
    def __init__(self, 
                 encoder,
                 num_classes,
                 hidden_dim=256,
                 dropout=0.1,
                 use_batch_norm=True):
        """
        Args:
            encoder: ProteinEncoder instance
            num_classes: Number of output classes
            hidden_dim: Hidden dimension for classification head
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(ProteinClassifier, self).__init__()
        
        self.encoder = encoder
        self.num_classes = num_classes
        
        # Classification head
        layers = []
        input_dim = encoder.hidden_nf // 4
        
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim // 2))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim // 2, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, node_features, edge_index, node_pos, edge_attr, batch=None):
        """
        Forward pass for classification.
        """
        # Get embeddings from encoder
        embeddings = self.encoder(
            node_features=node_features,
            edge_index=edge_index,
            node_pos=node_pos,
            edge_attr=edge_attr,
            batch=batch
        )
        
        # Apply classification head
        logits = self.classifier(embeddings)
        
        return logits


class ProteinRegressor(nn.Module):
    """
    Protein regressor for downstream regression tasks.
    """
    
    def __init__(self, 
                 encoder,
                 output_dim=1,
                 hidden_dim=256,
                 dropout=0.1,
                 use_batch_norm=True):
        """
        Args:
            encoder: ProteinEncoder instance
            output_dim: Number of output values (1 for single property prediction)
            hidden_dim: Hidden dimension for regression head
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(ProteinRegressor, self).__init__()
        
        self.encoder = encoder
        self.output_dim = output_dim
        
        # Regression head
        layers = []
        input_dim = encoder.hidden_nf // 4
        
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim // 2))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim // 2, output_dim))
        
        self.regressor = nn.Sequential(*layers)
    
    def forward(self, node_features, edge_index, node_pos, edge_attr, batch=None):
        """
        Forward pass for regression.
        """
        # Get embeddings from encoder
        embeddings = self.encoder(
            node_features=node_features,
            edge_index=edge_index,
            node_pos=node_pos,
            edge_attr=edge_attr,
            batch=batch
        )
        
        # Apply regression head
        outputs = self.regressor(embeddings)
        
        return outputs
