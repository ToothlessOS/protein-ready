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


class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

    def forward(self, batch_data):
        # Extract the two views directly from batch
        view1 = batch_data['view1']
        view2 = batch_data['view2']

        print(view1['edge_attr'].shape)
        
        # Forward pass through view1
        logits1, _ = self.model(
            node_features=view1['node_features'],
            edge_index=view1['edge_index'],
            node_pos=view1['node_pos'],
            edge_attr=view1['edge_attr'],
            batch=view1.get('batch', None)
        )
        
        # Forward pass through view2
        logits2, _ = self.model(
            node_features=view2['node_features'],
            edge_index=view2['edge_index'],
            node_pos=view2['node_pos'],
            edge_attr=view2['edge_attr'],
            batch=view2.get('batch', None)
        )
        
        return logits1, logits2

    def training_step(self, batch, batch_idx):
        # Skip batch if there was a load error
        if batch['load_error']:
            return None
            
        logits1, logits2 = self(batch)
        loss = self.loss_function(logits1, logits2)
        
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # Log learning rate
        opt = self.optimizers() if hasattr(self, "optimizers") else None
        if opt is not None:
            lr = opt.param_groups[0]['lr']
            self.log('lr', lr, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Skip batch if there was a load error
        if batch['load_error']:
            return None
            
        logits1, logits2 = self(batch)
        loss = self.loss_function(logits1, logits2)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'protein_id': batch['protein_id']}

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')

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
            elif self.hparams.lr_scheduler == 'cosine':
                warmup_epochs = self.hparams.lr_cosine_warmup_epochs
                max_epochs = self.hparams.max_epochs
                decay_ratio = self.hparams.lr_cosine_decay_ratio
                cycle_length = self.hparams.lr_cosine_cycle_length

                def combined_lr_lambda(epoch):
                    if epoch < warmup_epochs:
                        # Linear warm-up phase
                        return epoch / warmup_epochs
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
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'contrastive' or loss == 'ntxent':
            from .contrastive import ContrastiveLoss
            temperature = getattr(self.hparams, 'temperature', 0.07)
            self.loss_function = ContrastiveLoss(temperature=temperature)
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.hparams.model_name
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
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)