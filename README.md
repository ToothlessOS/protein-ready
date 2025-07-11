# Protein Representation Learning with Fine-tuning Support

A PyTorch Lightning implementation of protein representation learning using E(n) Equivariant Graph Neural Networks (EGNN) with contrastive learning, now enhanced with comprehensive fine-tuning support for downstream tasks.

## Features

### Core Capabilities
- **Protein Graph Representation**: Uses ESM embeddings (960D) as node features with 3D coordinate information
- **E(n) Equivariant Architecture**: EGNN layers that respect 3D rotational and translational symmetries
- **Contrastive Pretraining**: Self-supervised learning using protein subgraph augmentation
- **Downstream Task Support**: Easy fine-tuning for classification, regression, and multi-label tasks

### New Fine-tuning Features
- **Modular Architecture**: Separate encoder and task-specific heads
- **Flexible Fine-tuning**: Support for frozen backbone or end-to-end training
- **Multiple Task Types**: Classification, regression, and multi-label prediction
- **Two-stage Training**: Optional progressive fine-tuning strategy
- **Feature Extraction**: Easy embedding extraction for downstream analysis

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd protein-ready

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Pretraining (Original)

Train a protein representation model using contrastive learning:

```bash
python main.py \
    --batch_size 32 \
    --lr 1e-4 \
    --max_epochs 100 \
    --model_name contrastive \
    --loss contrastive \
    --data_path dataset/protein_g/ \
    --hidden_nf 512 \
    --egnn_layers 4
```

### 2. Fine-tuning for Classification

Fine-tune a pretrained model for protein classification:

```bash
python main.py \
    --downstream_task \
    --task_type classification \
    --num_classes 3 \
    --pretrained_path path/to/pretrained_model.ckpt \
    --freeze_backbone \
    --batch_size 16 \
    --lr 1e-4 \
    --max_epochs 50
```

### 3. Fine-tuning for Regression

Fine-tune for protein property prediction:

```bash
python main.py \
    --downstream_task \
    --task_type regression \
    --output_dim 1 \
    --pretrained_path path/to/pretrained_model.ckpt \
    --freeze_backbone \
    --batch_size 16 \
    --lr 1e-4 \
    --max_epochs 50
```

## Programming Interface

### Simple Fine-tuning Workflow

```python
from finetune_utils import create_simple_finetune_workflow

# Quick fine-tuning for classification
results = create_simple_finetune_workflow(
    pretrained_path="path/to/model.ckpt",
    task_type='classification',
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    num_classes=3,
    max_epochs=50
)
```

### Advanced Fine-tuning Control

```python
from finetune_utils import FineTuningWorkflow

# Create workflow
workflow = FineTuningWorkflow(
    pretrained_path="path/to/model.ckpt",
    task_type='classification',
    experiment_name='protein_classification'
)

# Configure model
model = workflow.create_model(
    num_classes=3,
    freeze_backbone=True,
    lr=1e-4
)

# Two-stage fine-tuning
results = workflow.fine_tune(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    test_dataloader=test_loader,
    two_stage_training=True,
    stage1_epochs=25,  # Frozen backbone
    stage2_epochs=25   # Unfrozen backbone
)
```

### Feature Extraction

```python
from model.protein_encoder import ProteinEncoder

# Load pretrained encoder
encoder = ProteinEncoder.from_pretrained("path/to/model.ckpt")

# Extract embeddings
embeddings = encoder(
    node_features=batch['node_features'],
    edge_index=batch['edge_index'],
    node_pos=batch['node_pos'],
    edge_attr=batch['edge_attr'],
    batch=batch['batch']
)
```

### Direct Model Creation

```python
from model.downstream_interface import create_downstream_model

# Create classification model
model = create_downstream_model(
    task_type='classification',
    pretrained_path="path/to/model.ckpt",
    num_classes=3,
    freeze_backbone=True
)

# Create regression model
model = create_downstream_model(
    task_type='regression',
    pretrained_path="path/to/model.ckpt",
    output_dim=1,
    freeze_backbone=True
)
```

## Examples

The `examples/` directory contains comprehensive examples:

- `classification_example.py`: Protein classification fine-tuning
- `regression_example.py`: Protein property regression and multi-property prediction

Run examples:
```bash
cd examples
python classification_example.py
python regression_example.py
```

## Architecture Overview

### Core Components

1. **ProteinEncoder**: The main encoder module with EGNN backbone
2. **ProteinClassifier**: Classification head for categorical predictions
3. **ProteinRegressor**: Regression head for continuous predictions
4. **DownstreamTaskInterface**: Lightning module for downstream tasks
5. **FineTuningWorkflow**: High-level workflow manager

### Model Architecture

```
Input Protein Graph
    ↓
ESM Embeddings (960D) + 3D Coordinates + Edge Features (11D)
    ↓
EGNN Layer 1 (960 → 512)
    ↓
EGNN Layer 2 (512 → 128)
    ↓
Graph Pooling (mean/max/sum)
    ↓
Task-Specific Head
    ↓
Output (Classification/Regression)
```

## Configuration

### Key Parameters

- `--hidden_nf`: Hidden dimension for EGNN layers (default: 512)
- `--egnn_layers`: Number of EGNN layers (default: 4)
- `--pooling`: Graph pooling method ('mean', 'max', 'sum')
- `--freeze_backbone`: Freeze pretrained encoder during fine-tuning
- `--temperature`: Temperature for contrastive loss (default: 0.07)

### Fine-tuning Strategies

1. **Frozen Backbone**: Train only task-specific head
2. **End-to-End**: Train entire model
3. **Two-Stage**: Progressive unfreezing (recommended)

## Data Format

Your protein data should include:
- `node_features`: ESM embeddings [num_nodes, 960]
- `edge_index`: Edge connectivity [2, num_edges]
- `node_pos`: 3D coordinates [num_nodes, 3]
- `edge_attr`: Edge features [num_edges, 11]
- `y`: Labels/targets for downstream tasks

## Results and Logging

- **TensorBoard Logs**: Saved in `fine_tuning_results/logs/`
- **Model Checkpoints**: Saved in `fine_tuning_results/checkpoints/`
- **Training Results**: Summary saved as text files

## Best Practices

### For Fine-tuning
1. Start with frozen backbone training
2. Use lower learning rates for pretrained parameters
3. Apply data augmentation carefully
4. Monitor validation metrics closely
5. Use early stopping to prevent overfitting

### For New Tasks
1. Analyze your data distribution
2. Choose appropriate pooling method
3. Adjust head architecture if needed
4. Consider multi-task learning for related tasks

## Troubleshooting

### Common Issues
- **CUDA Memory**: Reduce batch size or use gradient accumulation
- **Import Errors**: Ensure all dependencies are installed
- **Checkpoint Loading**: Verify checkpoint path and model compatibility

### Performance Tips
- Use appropriate batch sizes for your GPU memory
- Enable mixed precision training with `--precision 16`
- Use multiple workers for data loading

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{protein-ready,
  title={Protein Representation Learning with EGNN and Fine-tuning Support},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/protein-ready}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.