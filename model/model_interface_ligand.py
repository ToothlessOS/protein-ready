# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs

import pytorch_lightning as pl
import numpy as np


class MInterfaceLigand(pl.LightningModule):
    """
    PyTorch Lightning Module for protein-ligand contrastive learning.
    
    This module implements the MolCLR-style contrastive learning framework
    adapted for protein-ligand interactions. It supports GCN and GIN models
    for molecular graph representation learning.
    
    Based on MolCLR: https://github.com/yuyangw/MolCLR
    """
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

    def forward(self, batch_data):
        # Extract the two views from tuple format (DataBatch, DataBatch)
        view1, view2 = batch_data
        
        # Forward pass through view1 (expects a data object with .x, .edge_index, .edge_attr, .batch)
        # The GCN/GIN models expect data.x, data.edge_index, data.edge_attr, data.batch
        repr1, proj1 = self.model(view1)
        
        # Forward pass through view2
        repr2, proj2 = self.model(view2)
        
        return repr1, proj1, repr2, proj2

    def training_step(self, batch, batch_idx):
        repr1, proj1, repr2, proj2 = self(batch)
        
        # Use projections for contrastive loss (following MolCLR approach)
        # Normalize the projections as done in MolCLR
        proj1 = F.normalize(proj1, dim=1)
        proj2 = F.normalize(proj2, dim=1)
        
        loss = self.loss_function(proj1, proj2)
        
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        repr1, proj1, repr2, proj2 = self(batch)
        
        # Use projections for contrastive loss (following MolCLR approach)
        # Normalize the projections as done in MolCLR
        proj1 = F.normalize(proj1, dim=1)
        proj2 = F.normalize(proj2, dim=1)
        
        loss = self.loss_function(proj1, proj2)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Try to get protein_id from first DataBatch if it exists
        protein_id = 'unknown'
        if hasattr(batch[0], 'protein_id'):
            protein_id = batch[0].protein_id
        
        return {'val_loss': loss, 'protein_id': protein_id}

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Log learning rate at the end of each training batch."""
        # Get current learning rate from optimizer
        if hasattr(self, "trainer") and hasattr(self.trainer, "optimizers") and self.trainer.optimizers:
            optimizer = self.trainer.optimizers[0]
            current_lr = optimizer.param_groups[0]['lr']
            self.log('lr', current_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        """Log learning rate at the end of each validation epoch."""
        if hasattr(self, "trainer") and hasattr(self.trainer, "optimizers") and self.trainer.optimizers:
            optimizer = self.trainer.optimizers[0]
            current_lr = optimizer.param_groups[0]['lr']
            self.log('val_lr', current_lr, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            print('')

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'reduce-on-plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=self.hparams.lr_decay_ratio, patience=self.hparams.lr_decay_patience, min_lr=self.hparams.lr_decay_min_lr
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            elif self.hparams.lr_scheduler == 'cosine':
                warmup_epochs = self.hparams.lr_cosine_warmup_epochs
                max_epochs = self.hparams.max_epochs
                decay_ratio = self.hparams.lr_cosine_decay_ratio
                cycle_length = self.hparams.lr_cosine_cycle_length

                def combined_lr_lambda(epoch):
                    if epoch < warmup_epochs:
                        # Linear warm-up phase, avoid zero lr
                        min_lr_factor = 1e-6
                        return max(epoch / warmup_epochs, min_lr_factor)
                    else:
                        # After warm-up: combine decay with cyclic behavior
                        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
                    
                        # Exponential decay component
                        decay_factor = decay_ratio ** ((epoch - warmup_epochs) // 10)

                        # Cyclic component (triangular wave)
                        cycle_progress = ((epoch - warmup_epochs) % cycle_length) / cycle_length
                        cyclic_factor = 1.0 + 0.5 * np.sin(2 * np.pi * cycle_progress)
                    
                        return decay_factor * cyclic_factor
            
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, combined_lr_lambda)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            else:
                raise ValueError('Invalid lr_scheduler type!')

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'nt_xent':
            # Use the NT-Xent loss from MolCLR
            from .nt_xent_loss import NTXentLoss
            batch_size = getattr(self.hparams, 'batch_size', 32)
            temperature = getattr(self.hparams, 'temperature', 0.07)
            use_cosine_similarity = getattr(self.hparams, 'use_cosine_similarity', True)
            device = self.device
            self.loss_function = NTXentLoss(device, batch_size, temperature, use_cosine_similarity)
        else:
            raise ValueError(f"Invalid Loss Type: {loss}! Supported types: 'contrastive', 'ntxent', 'nt_xent', 'ntxent_complex'")

    def load_model(self):
        """
        Load the specified model architecture.
        
        Supported models:
        - 'gcn': Graph Convolutional Network from MolCLR
        - 'gin': Graph Isomorphism Network from MolCLR  
        - Other models following snake_case.py -> CamelCase convention
        """
        name = self.hparams.model_name
        # For GNN models (GCN, GIN), load from gnn.py
        if name.lower() in ['gcn', 'gin']:
            if name.lower() == 'gcn':
                from .gnn import GCN as Model
            elif name.lower() == 'gin':
                from .gnn import GINet as Model
            self.model = self.instancialize(Model)
        else:
            # Change the `snake_case.py` file name to `CamelCase` class name.
            # Please always name your model file name as `snake_case.py` and
            # class name corresponding `CamelCase`.
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
            try:
                Model = getattr(importlib.import_module(
                    '.'+name, package=__package__), camel_name)
            except:
                raise ValueError(
                    f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
            self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instantiate a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        try:
            # Use the newer inspect.signature for Python 3.3+
            sig = inspect.signature(Model.__init__)
            class_args = list(sig.parameters.keys())[1:]  # Skip 'self'
        except AttributeError:
            # Fallback for older Python versions
            class_args = inspect.getargspec(Model.__init__).args[1:]
        
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)

    def print_current_lr(self):
        """Print the current learning rate from the optimizer."""
        if hasattr(self, "trainer") and hasattr(self.trainer, "optimizers") and self.trainer.optimizers:
            optimizer = self.trainer.optimizers[0]
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr}")