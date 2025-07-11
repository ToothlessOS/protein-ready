"""
Downstream Task Interface for Fine-tuning Pretrained Protein Models

This module provides Lightning modules for various downstream tasks including
classification, regression, and other protein property prediction tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from scipy.stats import pearsonr, spearmanr
import numpy as np
from .protein_encoder import ProteinEncoder, ProteinClassifier, ProteinRegressor


class DownstreamTaskInterface(pl.LightningModule):
    """
    General Lightning module for downstream tasks using pretrained protein encoder.
    """
    
    def __init__(self, 
                 task_type='classification',  # 'classification', 'regression', 'multi_label'
                 num_classes=2,
                 output_dim=1,
                 pretrained_path=None,
                 freeze_backbone=True,
                 lr=1e-4,
                 weight_decay=1e-5,
                 lr_scheduler='cosine',
                 warmup_epochs=5,
                 max_epochs=100,
                 **encoder_kwargs):
        """
        Args:
            task_type: Type of downstream task
            num_classes: Number of classes for classification
            output_dim: Output dimension for regression
            pretrained_path: Path to pretrained model checkpoint
            freeze_backbone: Whether to freeze the pretrained encoder
            lr: Learning rate
            weight_decay: Weight decay for optimization
            lr_scheduler: Learning rate scheduler type
            warmup_epochs: Number of warmup epochs
            max_epochs: Maximum number of training epochs
            **encoder_kwargs: Additional arguments for the encoder
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.task_type = task_type
        self.num_classes = num_classes
        self.output_dim = output_dim
        
        # Initialize encoder
        if pretrained_path:
            self.encoder = ProteinEncoder.from_pretrained(
                pretrained_path, 
                freeze_backbone=freeze_backbone,
                **encoder_kwargs
            )
        else:
            self.encoder = ProteinEncoder(
                freeze_backbone=False,  # Train from scratch
                **encoder_kwargs
            )
        
        # Initialize task-specific head
        if task_type in ['classification', 'multi_label']:
            self.model = ProteinClassifier(
                encoder=self.encoder,
                num_classes=num_classes
            )
        elif task_type == 'regression':
            self.model = ProteinRegressor(
                encoder=self.encoder,
                output_dim=output_dim
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        # Define loss function
        if task_type == 'classification':
            self.loss_fn = nn.CrossEntropyLoss()
        elif task_type == 'multi_label':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif task_type == 'regression':
            self.loss_fn = nn.MSELoss()
    
    def forward(self, batch_data):
        """Forward pass through the model."""
        # Handle both single graph and batch data
        if isinstance(batch_data, dict):
            return self.model(
                node_features=batch_data['node_features'],
                edge_index=batch_data['edge_index'],
                node_pos=batch_data['node_pos'],
                edge_attr=batch_data['edge_attr'],
                batch=batch_data.get('batch', None)
            )
        else:
            # For PyTorch Geometric Data objects
            return self.model(
                node_features=batch_data.x,
                edge_index=batch_data.edge_index,
                node_pos=batch_data.pos,
                edge_attr=batch_data.edge_attr,
                batch=getattr(batch_data, 'batch', None)
            )
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        # Get predictions
        outputs = self.forward(batch)
        
        # Get targets
        if isinstance(batch, dict):
            targets = batch['y']
        else:
            targets = batch.y
        
        # Ensure correct shapes
        if self.task_type == 'classification':
            targets = targets.long()
        elif self.task_type in ['regression', 'multi_label']:
            targets = targets.float()
            if self.task_type == 'regression' and len(targets.shape) == 1:
                targets = targets.unsqueeze(-1)
        
        # Calculate loss
        loss = self.loss_fn(outputs, targets)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Task-specific metrics
        if self.task_type == 'classification':
            preds = torch.argmax(outputs, dim=1)
            acc = accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())
            self.log('train_acc', acc, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        outputs = self.forward(batch)
        
        # Get targets
        if isinstance(batch, dict):
            targets = batch['y']
        else:
            targets = batch.y
        
        # Ensure correct shapes
        if self.task_type == 'classification':
            targets = targets.long()
        elif self.task_type in ['regression', 'multi_label']:
            targets = targets.float()
            if self.task_type == 'regression' and len(targets.shape) == 1:
                targets = targets.unsqueeze(-1)
        
        # Calculate loss
        loss = self.loss_fn(outputs, targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return {
            'val_loss': loss,
            'outputs': outputs.detach(),
            'targets': targets.detach()
        }
    
    def validation_epoch_end(self, validation_step_outputs):
        """Compute validation metrics at the end of epoch."""
        # Concatenate all outputs and targets
        all_outputs = torch.cat([x['outputs'] for x in validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in validation_step_outputs])
        
        # Convert to numpy for sklearn metrics
        outputs_np = all_outputs.cpu().numpy()
        targets_np = all_targets.cpu().numpy()
        
        if self.task_type == 'classification':
            preds = np.argmax(outputs_np, axis=1)
            acc = accuracy_score(targets_np, preds)
            self.log('val_acc', acc, prog_bar=True)
            
            if self.num_classes == 2:
                # Binary classification - compute AUC
                probs = torch.softmax(all_outputs, dim=1)[:, 1].cpu().numpy()
                try:
                    auc = roc_auc_score(targets_np, probs)
                    self.log('val_auc', auc)
                except ValueError:
                    pass  # Skip AUC if only one class present
            
            # Precision, Recall, F1
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets_np, preds, average='weighted', zero_division=0
            )
            self.log('val_precision', precision)
            self.log('val_recall', recall)
            self.log('val_f1', f1)
        
        elif self.task_type == 'regression':
            # Pearson and Spearman correlation
            if len(outputs_np.shape) > 1 and outputs_np.shape[1] == 1:
                outputs_np = outputs_np.squeeze()
            if len(targets_np.shape) > 1 and targets_np.shape[1] == 1:
                targets_np = targets_np.squeeze()
            
            try:
                pearson_corr, _ = pearsonr(outputs_np, targets_np)
                spearman_corr, _ = spearmanr(outputs_np, targets_np)
                self.log('val_pearson', pearson_corr)
                self.log('val_spearman', spearman_corr)
            except ValueError:
                pass  # Skip correlation if constant values
            
            # MAE
            mae = np.mean(np.abs(outputs_np - targets_np))
            self.log('val_mae', mae)
    
    def test_step(self, batch, batch_idx):
        """Test step - same as validation."""
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, test_step_outputs):
        """Test epoch end - same as validation."""
        return self.validation_epoch_end(test_step_outputs)
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Separate parameters for backbone and head
        backbone_params = []
        head_params = []
        
        # Get backbone parameters (EGNN layers)
        for name, param in self.named_parameters():
            if 'encoder.egnn' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # Use different learning rates for backbone and head if backbone is not frozen
        if not self.encoder.freeze_backbone and len(backbone_params) > 0:
            param_groups = [
                {'params': backbone_params, 'lr': self.hparams.lr * 0.1},  # Lower LR for pretrained backbone
                {'params': head_params, 'lr': self.hparams.lr}
            ]
        else:
            param_groups = [{'params': head_params, 'lr': self.hparams.lr}]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay
        )
        
        if self.hparams.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.hparams.max_epochs,
                eta_min=self.hparams.lr * 0.01
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        elif self.hparams.lr_scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                min_lr=self.hparams.lr * 0.001
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch'
                }
            }
        else:
            return optimizer
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone for end-to-end fine-tuning."""
        self.encoder.unfreeze_backbone()
        print("Backbone unfrozen. Model ready for end-to-end fine-tuning.")


def create_downstream_model(task_type, pretrained_path=None, **kwargs):
    """
    Factory function to create downstream task models.
    
    Args:
        task_type: Type of task ('classification', 'regression', 'multi_label')
        pretrained_path: Path to pretrained model checkpoint
        **kwargs: Additional arguments for the model
    
    Returns:
        DownstreamTaskInterface instance
    """
    return DownstreamTaskInterface(
        task_type=task_type,
        pretrained_path=pretrained_path,
        **kwargs
    )
