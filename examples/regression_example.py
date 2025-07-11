"""
Example: Fine-tuning Pretrained Protein Model for Regression

This example demonstrates how to fine-tune the pretrained protein representation
model for a downstream regression task (e.g., predicting protein stability, binding affinity, etc.).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
from pathlib import Path

# Import the fine-tuning utilities
from model.downstream_interface import DownstreamTaskInterface, create_downstream_model
from model.protein_encoder import ProteinEncoder
from finetune_utils import FineTuningWorkflow, extract_embeddings


def create_dummy_regression_data(num_samples=1000):
    """
    Create dummy protein regression data for demonstration.
    In practice, you would load your actual protein data with continuous targets.
    """
    data = []
    targets = []
    
    for i in range(num_samples):
        # Simulate protein graph data
        num_nodes = np.random.randint(20, 100)
        
        # Node features (ESM embeddings)
        node_features = torch.randn(num_nodes, 960)
        
        # Node positions (3D coordinates)
        node_pos = torch.randn(num_nodes, 3)
        
        # Edge indices (fully connected for simplicity)
        edge_index = torch.combinations(torch.arange(num_nodes), 2).T
        
        # Edge attributes
        num_edges = edge_index.shape[1]
        edge_attr = torch.randn(num_edges, 11)
        
        # Simulate regression target (e.g., stability score, binding affinity)
        # Make it somewhat correlated with protein size for realism
        target = np.random.normal(0, 1) + 0.1 * np.log(num_nodes)
        
        sample = {
            'node_features': node_features,
            'edge_index': edge_index,
            'node_pos': node_pos,
            'edge_attr': edge_attr,
            'y': torch.tensor([target], dtype=torch.float32)  # Regression target
        }
        
        data.append(sample)
        targets.append(target)
    
    return data, targets


def collate_fn(batch):
    """Custom collate function for batching protein graphs."""
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
        batch_data['node_features'].append(sample['node_features'])
        batch_data['node_pos'].append(sample['node_pos'])
        
        edge_index = sample['edge_index'] + node_offset
        batch_data['edge_index'].append(edge_index)
        batch_data['edge_attr'].append(sample['edge_attr'])
        batch_data['y'].append(sample['y'])
        
        num_nodes = sample['node_features'].shape[0]
        batch_data['batch'].extend([i] * num_nodes)
        node_offset += num_nodes
    
    return {
        'node_features': torch.cat(batch_data['node_features'], dim=0),
        'edge_index': torch.cat(batch_data['edge_index'], dim=1),
        'node_pos': torch.cat(batch_data['node_pos'], dim=0),
        'edge_attr': torch.cat(batch_data['edge_attr'], dim=0),
        'y': torch.stack(batch_data['y']),
        'batch': torch.tensor(batch_data['batch'])
    }


def example_regression_finetuning():
    """Example of fine-tuning for protein property regression."""
    print("=== Protein Regression Fine-tuning Example ===\n")
    
    # Parameters
    output_dim = 1  # Single property prediction
    num_samples = 500
    batch_size = 8
    max_epochs = 30
    
    # Create dummy data
    print("Creating dummy protein regression data...")
    data, targets = create_dummy_regression_data(num_samples)
    
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
    print(f"Target statistics - Mean: {np.mean(targets):.3f}, Std: {np.std(targets):.3f}")
    
    # Pretrained model path
    pretrained_path = "path/to/your/pretrained_model.ckpt"
    if not Path(pretrained_path).exists():
        print(f"WARNING: Pretrained model not found at {pretrained_path}")
        print("Training from scratch for demonstration...")
        pretrained_path = None
    
    # Fine-tuning workflow
    print("\n--- Fine-tuning for Regression ---")
    try:
        workflow = FineTuningWorkflow(
            pretrained_path=pretrained_path,
            task_type='regression',
            experiment_name='protein_regression'
        )
        
        # Create model with regression configuration
        model = workflow.create_model(
            output_dim=output_dim,
            freeze_backbone=True,
            lr=1e-4,
            weight_decay=1e-5
        )
        
        # Create trainer with regression-specific monitoring
        trainer = workflow.create_trainer(
            max_epochs=max_epochs,
            monitor_metric='val_loss',  # Monitor validation loss for regression
            patience=10
        )
        
        # Fine-tune with two-stage training
        results = workflow.fine_tune(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            two_stage_training=True,
            stage1_epochs=max_epochs // 2,
            stage2_epochs=max_epochs // 2
        )
        
        print("Regression fine-tuning completed!")
        print("\nFinal Results:")
        for stage, metrics in results.items():
            print(f"\n{stage.upper()}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # Load best model and make predictions
        best_model = workflow.load_best_model()
        
        # Extract embeddings for analysis
        print("\n--- Extracting Embeddings ---")
        embeddings, labels = extract_embeddings(best_model, test_loader)
        print(f"Extracted embeddings shape: {embeddings.shape}")
        print(f"Embedding statistics - Mean: {embeddings.mean():.3f}, Std: {embeddings.std():.3f}")
        
    except Exception as e:
        print(f"Regression workflow failed: {e}")
        import traceback
        traceback.print_exc()


def example_multi_property_regression():
    """Example of multi-property regression (predicting multiple properties simultaneously)."""
    print("\n=== Multi-Property Regression Example ===\n")
    
    def create_multi_property_data(num_samples=500, num_properties=3):
        """Create data with multiple regression targets."""
        data = []
        
        for i in range(num_samples):
            num_nodes = np.random.randint(20, 100)
            
            node_features = torch.randn(num_nodes, 960)
            node_pos = torch.randn(num_nodes, 3)
            edge_index = torch.combinations(torch.arange(num_nodes), 2).T
            edge_attr = torch.randn(edge_index.shape[1], 11)
            
            # Multiple correlated properties
            base_score = np.random.normal(0, 1)
            properties = []
            for j in range(num_properties):
                # Make properties somewhat correlated with some noise
                prop = base_score + np.random.normal(0, 0.5) + 0.1 * j
                properties.append(prop)
            
            sample = {
                'node_features': node_features,
                'edge_index': edge_index,
                'node_pos': node_pos,
                'edge_attr': edge_attr,
                'y': torch.tensor(properties, dtype=torch.float32)
            }
            
            data.append(sample)
        
        return data
    
    # Parameters
    num_properties = 3
    output_dim = num_properties
    num_samples = 300
    batch_size = 6
    
    # Create multi-property data
    print(f"Creating multi-property data ({num_properties} properties)...")
    data = create_multi_property_data(num_samples, num_properties)
    
    # Split data
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"Dataset sizes - Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Create and train multi-property model
    try:
        model = create_downstream_model(
            task_type='regression',
            pretrained_path=None,  # Train from scratch for demo
            output_dim=output_dim,
            freeze_backbone=False,
            lr=1e-4
        )
        
        trainer = pl.Trainer(
            max_epochs=20,
            accelerator='auto',
            devices='auto',
            enable_progress_bar=True
        )
        
        trainer.fit(model, train_loader, val_loader)
        
        print("Multi-property regression completed!")
        
        # Test predictions
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                predictions = model(batch)
                targets = batch['y']
                
                print(f"\nBatch predictions shape: {predictions.shape}")
                print(f"Sample prediction: {predictions[0].cpu().numpy()}")
                print(f"Sample target: {targets[0].cpu().numpy()}")
                break
        
    except Exception as e:
        print(f"Multi-property regression failed: {e}")


if __name__ == "__main__":
    # Run single-property regression example
    example_regression_finetuning()
    
    # Run multi-property regression example  
    example_multi_property_regression()
    
    print("\n=== Regression examples completed! ===")
