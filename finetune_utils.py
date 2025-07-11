"""
Fine-tuning Utilities for Protein Representation Models

This module provides high-level utilities and workflows for fine-tuning
pretrained protein models on downstream tasks.
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from model.downstream_interface import DownstreamTaskInterface
from model.protein_encoder import ProteinEncoder


class FineTuningWorkflow:
    """
    High-level workflow manager for fine-tuning pretrained protein models.
    """
    
    def __init__(self, 
                 pretrained_path: str,
                 task_type: str = 'classification',
                 experiment_name: str = 'protein_finetune',
                 output_dir: str = './fine_tuning_results',
                 seed: int = 42):
        """
        Args:
            pretrained_path: Path to pretrained model checkpoint
            task_type: Type of downstream task ('classification', 'regression', 'multi_label')
            experiment_name: Name for the experiment
            output_dir: Directory to save results
            seed: Random seed for reproducibility
        """
        self.pretrained_path = pretrained_path
        self.task_type = task_type
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        # Set random seeds
        pl.seed_everything(seed)
        
        self.model = None
        self.trainer = None
        self.results = {}
    
    def create_model(self, 
                    num_classes: Optional[int] = None,
                    output_dim: Optional[int] = None,
                    freeze_backbone: bool = True,
                    lr: float = 1e-4,
                    weight_decay: float = 1e-5,
                    **model_kwargs) -> DownstreamTaskInterface:
        """
        Create and configure the downstream model.
        
        Args:
            num_classes: Number of classes for classification tasks
            output_dim: Output dimension for regression tasks
            freeze_backbone: Whether to freeze the pretrained backbone
            lr: Learning rate
            weight_decay: Weight decay
            **model_kwargs: Additional model arguments
        
        Returns:
            Configured DownstreamTaskInterface model
        """
        model_config = {
            'task_type': self.task_type,
            'pretrained_path': self.pretrained_path,
            'freeze_backbone': freeze_backbone,
            'lr': lr,
            'weight_decay': weight_decay,
            **model_kwargs
        }
        
        if self.task_type in ['classification', 'multi_label'] and num_classes is not None:
            model_config['num_classes'] = num_classes
        elif self.task_type == 'regression' and output_dim is not None:
            model_config['output_dim'] = output_dim
        
        self.model = DownstreamTaskInterface(**model_config)
        return self.model
    
    def create_trainer(self,
                      max_epochs: int = 50,
                      patience: int = 10,
                      monitor_metric: str = 'val_loss',
                      accelerator: str = 'auto',
                      devices: str = 'auto',
                      precision: str = '32-true',
                      **trainer_kwargs) -> pl.Trainer:
        """
        Create and configure the PyTorch Lightning trainer.
        
        Args:
            max_epochs: Maximum number of training epochs
            patience: Early stopping patience
            monitor_metric: Metric to monitor for early stopping
            accelerator: Accelerator type
            devices: Device configuration
            precision: Training precision
            **trainer_kwargs: Additional trainer arguments
        
        Returns:
            Configured PyTorch Lightning Trainer
        """
        # Setup callbacks
        callbacks = []
        
        # Model checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.output_dir / 'checkpoints',
            filename=f'{self.experiment_name}-{{epoch:02d}}-{{{monitor_metric}:.3f}}',
            monitor=monitor_metric,
            mode='min' if 'loss' in monitor_metric else 'max',
            save_top_k=3,
            save_last=True,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if patience > 0:
            early_stop_callback = EarlyStopping(
                monitor=monitor_metric,
                patience=patience,
                mode='min' if 'loss' in monitor_metric else 'max',
                verbose=True
            )
            callbacks.append(early_stop_callback)
        
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)
        
        # Logger
        logger = TensorBoardLogger(
            save_dir=self.output_dir / 'logs',
            name=self.experiment_name,
            version=None
        )
        
        # Create trainer
        trainer_config = {
            'max_epochs': max_epochs,
            'accelerator': accelerator,
            'devices': devices,
            'precision': precision,
            'callbacks': callbacks,
            'logger': logger,
            'enable_progress_bar': True,
            'enable_model_summary': True,
            **trainer_kwargs
        }
        
        self.trainer = pl.Trainer(**trainer_config)
        return self.trainer
    
    def fine_tune(self,
                  train_dataloader: DataLoader,
                  val_dataloader: DataLoader,
                  test_dataloader: Optional[DataLoader] = None,
                  two_stage_training: bool = True,
                  stage1_epochs: int = 20,
                  stage2_epochs: int = 30) -> Dict[str, Any]:
        """
        Execute the fine-tuning workflow.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            test_dataloader: Optional test data loader
            two_stage_training: Whether to use two-stage training (frozen then unfrozen)
            stage1_epochs: Epochs for stage 1 (frozen backbone)
            stage2_epochs: Epochs for stage 2 (unfrozen backbone)
        
        Returns:
            Dictionary containing training results and metrics
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        results = {}
        
        if two_stage_training:
            print("=== Starting Two-Stage Fine-tuning ===")
            
            # Stage 1: Train with frozen backbone
            print("\n--- Stage 1: Training with frozen backbone ---")
            self.create_trainer(max_epochs=stage1_epochs)
            self.trainer.fit(self.model, train_dataloader, val_dataloader)
            
            # Save stage 1 results
            stage1_metrics = self.trainer.callback_metrics
            results['stage1_metrics'] = {k: float(v) for k, v in stage1_metrics.items()}
            
            # Stage 2: Unfreeze and continue training
            print("\n--- Stage 2: Training with unfrozen backbone ---")
            self.model.unfreeze_backbone()
            
            # Create new trainer for stage 2 with lower learning rate
            self.create_trainer(max_epochs=stage2_epochs)
            self.trainer.fit(
                self.model, 
                train_dataloader, 
                val_dataloader,
                ckpt_path=self.trainer.checkpoint_callback.best_model_path
            )
            
            # Save stage 2 results
            stage2_metrics = self.trainer.callback_metrics
            results['stage2_metrics'] = {k: float(v) for k, v in stage2_metrics.items()}
            
        else:
            print("=== Starting Single-Stage Fine-tuning ===")
            self.create_trainer(max_epochs=stage1_epochs + stage2_epochs)
            self.trainer.fit(self.model, train_dataloader, val_dataloader)
            
            # Save results
            final_metrics = self.trainer.callback_metrics
            results['final_metrics'] = {k: float(v) for k, v in final_metrics.items()}
        
        # Test evaluation if test dataloader provided
        if test_dataloader is not None:
            print("\n--- Evaluating on test set ---")
            test_results = self.trainer.test(self.model, test_dataloader, verbose=True)
            results['test_metrics'] = test_results[0] if test_results else {}
        
        # Save results
        self.results = results
        self._save_results()
        
        return results
    
    def _save_results(self):
        """Save training results to file."""
        results_file = self.output_dir / f'{self.experiment_name}_results.txt'
        
        with open(results_file, 'w') as f:
            f.write(f"Fine-tuning Results for {self.experiment_name}\n")
            f.write("=" * 50 + "\n\n")
            
            for stage, metrics in self.results.items():
                f.write(f"{stage.upper()}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")
        
        print(f"Results saved to {results_file}")
    
    def load_best_model(self) -> DownstreamTaskInterface:
        """
        Load the best model from checkpoints.
        
        Returns:
            Best trained model
        """
        if self.trainer is None:
            raise ValueError("No trainer found. Run fine_tune() first.")
        
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        if best_model_path:
            model = DownstreamTaskInterface.load_from_checkpoint(best_model_path)
            return model
        else:
            return self.model


def extract_embeddings(model: DownstreamTaskInterface,
                      dataloader: DataLoader,
                      device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings from a trained model.
    
    Args:
        model: Trained downstream model
        dataloader: Data loader for extraction
        device: Device to run inference on
    
    Returns:
        Tuple of (embeddings, labels)
    """
    model.eval()
    model.to(device)
    
    embeddings_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            if isinstance(batch, dict):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                labels = batch['y']
            else:
                batch = batch.to(device)
                labels = batch.y
            
            # Get embeddings from encoder
            embeddings = model.encoder(
                node_features=batch['node_features'] if isinstance(batch, dict) else batch.x,
                edge_index=batch['edge_index'] if isinstance(batch, dict) else batch.edge_index,
                node_pos=batch['node_pos'] if isinstance(batch, dict) else batch.pos,
                edge_attr=batch['edge_attr'] if isinstance(batch, dict) else batch.edge_attr,
                batch=batch.get('batch', None) if isinstance(batch, dict) else getattr(batch, 'batch', None)
            )
            
            embeddings_list.append(embeddings.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    
    embeddings = np.vstack(embeddings_list)
    labels = np.concatenate(labels_list)
    
    return embeddings, labels


def create_simple_finetune_workflow(pretrained_path: str,
                                   task_type: str,
                                   train_dataloader: DataLoader,
                                   val_dataloader: DataLoader,
                                   num_classes: Optional[int] = None,
                                   output_dim: Optional[int] = None,
                                   max_epochs: int = 50,
                                   experiment_name: str = 'protein_finetune') -> Dict[str, Any]:
    """
    Simple one-function workflow for fine-tuning.
    
    Args:
        pretrained_path: Path to pretrained model
        task_type: Type of task
        train_dataloader: Training data
        val_dataloader: Validation data
        num_classes: Number of classes (for classification)
        output_dim: Output dimension (for regression)
        max_epochs: Maximum training epochs
        experiment_name: Name for the experiment
    
    Returns:
        Training results
    """
    workflow = FineTuningWorkflow(
        pretrained_path=pretrained_path,
        task_type=task_type,
        experiment_name=experiment_name
    )
    
    # Create model
    model_kwargs = {}
    if num_classes is not None:
        model_kwargs['num_classes'] = num_classes
    if output_dim is not None:
        model_kwargs['output_dim'] = output_dim
    
    workflow.create_model(**model_kwargs)
    
    # Fine-tune
    results = workflow.fine_tune(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        stage1_epochs=max_epochs // 2,
        stage2_epochs=max_epochs // 2
    )
    
    return results
