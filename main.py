# Copyright 2021 Zhongyang Zhang
# Contact: mirakuruyoo@gmai.com
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

""" This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and 
    DInterface can be seen as transparent to all your args.    
"""
import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.cli import LightningArgumentParser

from model import MInterface
from data import DInterface
from utils import load_model_path_by_args

# Import downstream task interface for fine-tuning
try:
    from model.downstream_interface import DownstreamTaskInterface
    DOWNSTREAM_AVAILABLE = True
except ImportError:
    DOWNSTREAM_AVAILABLE = False
    print("Downstream interface not available. Only pretraining mode supported.")


def load_callbacks():
    callbacks = []
    """
    callbacks.append(plc.EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=10,
        min_delta=0.001
    ))
    """

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_loss',
        filename='best-{epoch:03d}-{val_loss:.3f}',
        save_top_k=1,
        mode='min',
        save_last=True
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks


def main(args):
    pl.seed_everything(args.seed)
    
    # Check if this is a downstream task
    is_downstream = hasattr(args, 'downstream_task') and args.downstream_task
    
    if is_downstream and DOWNSTREAM_AVAILABLE:
        # Downstream task mode
        print(f"Running downstream task: {args.task_type}")
        
        model = DownstreamTaskInterface(
            task_type=args.task_type,
            num_classes=getattr(args, 'num_classes', 2),
            output_dim=getattr(args, 'output_dim', 1),
            pretrained_path=getattr(args, 'pretrained_path', None),
            freeze_backbone=getattr(args, 'freeze_backbone', True),
            lr=args.lr,
            weight_decay=args.weight_decay,
            lr_scheduler=args.lr_scheduler,
            max_epochs=args.max_epochs
        )
        data_module = DInterface(**vars(args))
        
    else:
        # Original pretraining mode
        load_path = load_model_path_by_args(args)
        data_module = DInterface(**vars(args))

        if load_path is None:
            model = MInterface(**vars(args))
        else:
            model = MInterface(**vars(args))
            args.ckpt_path = load_path

    # # If you want to change the logger's saving folder
    # logger = TensorBoardLogger(save_dir='kfold_log', name=args.log_dir)
    callbacks = load_callbacks()
    # args.logger = logger

    # Modern PyTorch Lightning trainer initialization
    trainer = Trainer(
        max_epochs=getattr(args, 'max_epochs', 100),
        accelerator=getattr(args, 'accelerator', 'auto'),
        devices=getattr(args, 'devices', 'auto'),
        callbacks=callbacks,
        logger=getattr(args, 'logger', True),
        check_val_every_n_epoch=getattr(args, 'check_val_every_n_epoch', 1),
        val_check_interval=getattr(args, 'val_check_interval', 1.0),
        num_sanity_val_steps=getattr(args, 'num_sanity_val_steps', 2),
        log_every_n_steps=getattr(args, 'log_every_n_steps', 1),
        precision=getattr(args, 'precision', '32-true'),
        fast_dev_run=getattr(args, 'fast_dev_run', False),
        limit_train_batches=getattr(args, 'limit_train_batches', 1.0),
        limit_val_batches=getattr(args, 'limit_val_batches', 1.0),
        limit_test_batches=getattr(args, 'limit_test_batches', 1.0),
    )
    
    ckpt_path = getattr(args, 'ckpt_path', None)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)


if __name__ == '__main__':
    parser = LightningArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['reduce-on-plateau', 'cosine'], type=str, default='cosine')
    parser.add_argument('--lr_decay_ratio', default=0.95, type=float)
    parser.add_argument('--lr_decay_patience', default=4, type=int)
    parser.add_argument('--lr_cosine_warmup_epochs', default=10, type=int)
    parser.add_argument('--lr_cosine_cycle_length', default=20, type=float)
    parser.add_argument('--lr_cosine_decay_ratio', default=0.99, type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Training Info
    parser.add_argument('--dataset', default='protein_dataset', type=str)
    parser.add_argument('--data_path', default='dataset/protein_g/', type=str)
    parser.add_argument('--complete_graph_percent', default=0, type=int, help='Percentage of complete graphs in the dataset') # Different from GearNet
    parser.add_argument('--test_percent', default=0.01, type=float, help='Percentage of data used for testing')
    parser.add_argument('--model_name', default='contrastive', type=str)
    parser.add_argument('--loss', default='contrastive', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='lightning_logs', type=str)
    
    # Downstream task arguments
    parser.add_argument('--downstream_task', action='store_true', help='Enable downstream task mode')
    parser.add_argument('--task_type', default='classification', choices=['classification', 'regression', 'multi_label'], help='Type of downstream task')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of classes for classification tasks')
    parser.add_argument('--output_dim_downstream', default=1, type=int, help='Output dimension for regression tasks')
    parser.add_argument('--pretrained_path', default=None, type=str, help='Path to pretrained model checkpoint')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze pretrained backbone during fine-tuning')
    
    # Model Hyperparameters
    parser.add_argument('--in_node_nf', default=960, type=int, help='Input node feature dimension (ESM embeddings)')
    parser.add_argument('--in_edge_nf', default=11, type=int, help='Input edge feature dimension')
    parser.add_argument('--hidden_nf', default=512, type=int, help='Hidden dimension for EGNN')
    parser.add_argument('--egnn_layers', default=4, type=int, help='Number of EGNN layers')
    parser.add_argument('--output_dim', default=64, type=int, help='Final output dimension')
    parser.add_argument('--projection_dim', default=96, type=int, help='Projection dimension for contrastive learning')
    parser.add_argument('--pooling', default='mean', choices=['mean', 'max', 'sum'], help='Graph pooling method')
    parser.add_argument('--temperature', default=0.07, type=float, help='Temperature for contrastive loss')
    
    # Subgraph sampling parameters
    parser.add_argument('--min_nodes', default=10, type=int, help='Minimum number of nodes in subgraph')
    parser.add_argument('--max_nodes', default=100, type=int, help='Maximum number of nodes in subgraph')
    
    # Trainer arguments
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--accelerator', default='auto', type=str)
    parser.add_argument('--devices', default='auto')
    parser.add_argument('--precision', default='32-true', type=str)
    parser.add_argument('--fast_dev_run', action='store_true')
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int, help='Run validation every N epochs')
    parser.add_argument('--val_check_interval', default=1.0, type=float, help='Validation check interval (fraction of epoch or steps)')
    parser.add_argument('--num_sanity_val_steps', default=2, type=int, help='Number of validation steps to run before training')
    parser.add_argument('--log_every_n_steps', default=1, type=int, help='Log every N steps (set to 1 for small datasets)')
    
    # Dataset and caching parameters
    parser.add_argument('--cache_size', default=512, type=int, help='Cache size for dataset')
    parser.add_argument('--enable_cache', action='store_true', help='Enable dataset caching')
    parser.add_argument('--preload_cache', action='store_true', help='Preload cache for dataset')

    args = parser.parse_args()

    main(args)
