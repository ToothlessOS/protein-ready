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