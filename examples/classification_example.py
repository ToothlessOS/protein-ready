"""
Example: Fine-tuning Pretrained Protein Model for Classification

This example demonstrates how to fine-tune the pretrained protein representation
model for a downstream classification task.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import numpy as np
from pathlib import Path

# Import the fine-tuning utilities
from model.downstream_interface import DownstreamTaskInterface, create_downstream_model
from model.protein_encoder import ProteinEncoder
from finetune_utils import FineTuningWorkflow, create_simple_finetune_workflow


def create_dummy_protein_data(num_samples=1000, num_classes=3):
    """
    Create dummy protein data for demonstration.
    In practice, you would load your actual protein graph data.
    """
    data = []
    labels = []
    
    for i in range(num_samples):
        # Simulate protein graph data
        num_nodes = np.random.randint(20, 100)  # Variable protein size
        
        # Node features (ESM embeddings)
        node_features = torch.randn(num_nodes, 960)
        
        # Node positions (3D coordinates)
        node_pos = torch.randn(num_nodes, 3)
        
        # Edge indices (fully connected for simplicity)
        edge_index = torch.combinations(torch.arange(num_nodes), 2).T
        
        # Edge attributes
        num_edges = edge_index.shape[1]
        edge_attr = torch.randn(num_edges, 11)
        
        # Random label
        label = torch.randint(0, num_classes, (1,)).item()
        
        sample = {
            'node_features': node_features,
            'edge_index': edge_index,
            'node_pos': node_pos,
            'edge_attr': edge_attr,
            'y': torch.tensor(label)
        }
        
        data.append(sample)
        labels.append(label)
    
    return data, labels


def collate_fn(batch):
    """
    Custom collate function for batching protein graphs.
    """
    batch_data = {
        'node_features': [],
        'edge_index': [],
        'node_pos': [],
        'edge_attr': [],
        'y': [],
        'batch': []
    }
    
    node_offset = 0
    for i, sample in enumerate(batch):
        # Add node features and positions
        batch_data['node_features'].append(sample['node_features'])
        batch_data['node_pos'].append(sample['node_pos'])
        
        # Adjust edge indices for batching
        edge_index = sample['edge_index'] + node_offset
        batch_data['edge_index'].append(edge_index)
        
        # Add edge attributes
        batch_data['edge_attr'].append(sample['edge_attr'])
        
        # Add labels
        batch_data['y'].append(sample['y'])
        
        # Create batch index
        num_nodes = sample['node_features'].shape[0]
        batch_data['batch'].extend([i] * num_nodes)
        
        node_offset += num_nodes
    
    # Concatenate all data
    return {
        'node_features': torch.cat(batch_data['node_features'], dim=0),
        'edge_index': torch.cat(batch_data['edge_index'], dim=1),
        'node_pos': torch.cat(batch_data['node_pos'], dim=0),
        'edge_attr': torch.cat(batch_data['edge_attr'], dim=0),
        'y': torch.stack(batch_data['y']),
        'batch': torch.tensor(batch_data['batch'])
    }


def example_classification_finetuning():
    """
    Example of fine-tuning for protein classification.
    """
    print("=== Protein Classification Fine-tuning Example ===\n")
    
    # Parameters
    num_classes = 3
    num_samples = 500
    batch_size = 8
    max_epochs = 20
    
    # Create dummy data (replace with your actual data loading)
    print("Creating dummy protein data...")
    data, labels = create_dummy_protein_data(num_samples, num_classes)
    
    # Split data
    train_size = int(0.7 * len(data))
    val_size = int(0.15 * len(data))
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"Dataset sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Pretrained model path (update with your actual path)
    pretrained_path = "path/to/your/pretrained_model.ckpt"
    
    # Check if pretrained model exists
    if not Path(pretrained_path).exists():
        print(f"WARNING: Pretrained model not found at {pretrained_path}")
        print("Creating model from scratch for demonstration...")
        pretrained_path = None
    
    # Method 1: Using the simple workflow function
    print("\n--- Method 1: Simple Workflow ---")
    try:
        results = create_simple_finetune_workflow(
            pretrained_path=pretrained_path,
            task_type='classification',
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_classes=num_classes,
            max_epochs=max_epochs,
            experiment_name='protein_classification_simple'
        )
        print("Simple workflow completed successfully!")
        print("Results:", results)
    except Exception as e:
        print(f"Simple workflow failed: {e}")
    
    # Method 2: Using the detailed workflow class
    print("\n--- Method 2: Detailed Workflow ---")
    try:
        workflow = FineTuningWorkflow(
            pretrained_path=pretrained_path,
            task_type='classification',
            experiment_name='protein_classification_detailed'
        )
        
        # Create model
        model = workflow.create_model(
            num_classes=num_classes,
            freeze_backbone=True,
            lr=1e-4
        )
        
        # Create trainer
        trainer = workflow.create_trainer(
            max_epochs=max_epochs,
            patience=5
        )
        
        # Fine-tune
        results = workflow.fine_tune(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            two_stage_training=True,
            stage1_epochs=max_epochs // 2,
            stage2_epochs=max_epochs // 2
        )
        
        print("Detailed workflow completed successfully!")
        print("Results:", results)
        
        # Load best model
        best_model = workflow.load_best_model()
        print("Best model loaded successfully!")
        
    except Exception as e:
        print(f"Detailed workflow failed: {e}")
    
    # Method 3: Manual approach for full control
    print("\n--- Method 3: Manual Approach ---")
    try:
        # Create model directly
        model = create_downstream_model(
            task_type='classification',
            pretrained_path=pretrained_path,
            num_classes=num_classes,
            freeze_backbone=True,
            lr=1e-4
        )
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator='auto',
            devices='auto',
            enable_progress_bar=True
        )
        
        # Train
        trainer.fit(model, train_loader, val_loader)
        
        # Test
        if test_loader:
            trainer.test(model, test_loader)
        
        print("Manual approach completed successfully!")
        
    except Exception as e:
        print(f"Manual approach failed: {e}")


def example_feature_extraction():
    """
    Example of using the pretrained model for feature extraction.
    """
    print("\n=== Feature Extraction Example ===\n")
    
    # Create dummy protein data
    data, _ = create_dummy_protein_data(num_samples=10)
    data_loader = DataLoader(data, batch_size=4, collate_fn=collate_fn)
    
    # Load pretrained encoder (or create from scratch for demo)
    try:
        encoder = ProteinEncoder.from_pretrained("path/to/pretrained_model.ckpt")
        print("Loaded pretrained encoder")
    except:
        encoder = ProteinEncoder()
        print("Created encoder from scratch")
    
    # Extract features
    encoder.eval()
    with torch.no_grad():
        for batch in data_loader:
            # Get embeddings
            embeddings = encoder(
                node_features=batch['node_features'],
                edge_index=batch['edge_index'],
                node_pos=batch['node_pos'],
                edge_attr=batch['edge_attr'],
                batch=batch['batch']
            )
            
            print(f"Batch size: {len(torch.unique(batch['batch']))}")
            print(f"Embedding shape: {embeddings.shape}")
            print(f"Embedding statistics - Mean: {embeddings.mean():.3f}, Std: {embeddings.std():.3f}")
            break


if __name__ == "__main__":
    # Run classification fine-tuning example
    example_classification_finetuning()
    
    # Run feature extraction example
    example_feature_extraction()
    
    print("\n=== Examples completed! ===")
