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
        enable_checkpointing=getattr(args, 'enable_checkpointing', True),
        enable_progress_bar=getattr(args, 'enable_progress_bar', True),
        enable_model_summary=getattr(args, 'enable_model_summary', True),
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
    parser.add_argument('--lr', default=1e-3, type=float)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Training Info
    parser.add_argument('--dataset', default='protein_dataset', type=str)
    parser.add_argument('--data_path', default='dataset/protein_g/', type=str)
    parser.add_argument('--model_name', default='contrastive', type=str)
    parser.add_argument('--loss', default='contrastive', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='lightning_logs', type=str)
    
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
    parser.add_argument('--limit_train_batches', default=1.0, type=float)
    parser.add_argument('--limit_val_batches', default=1.0, type=float)
    parser.add_argument('--limit_test_batches', default=1.0, type=float)
    
    # Legacy parameters (kept for compatibility)
    parser.add_argument('--hid', default=64, type=int)
    parser.add_argument('--block_num', default=8, type=int)
    parser.add_argument('--in_channel', default=3, type=int)
    parser.add_argument('--layer_num', default=5, type=int)

    # Other
    parser.add_argument('--aug_prob', default=0.5, type=float)
    
    # Dataset and caching parameters
    parser.add_argument('--cache_size', default=512, type=int, help='Cache size for dataset')
    parser.add_argument('--enable_cache', action='store_true', help='Enable dataset caching')
    parser.add_argument('--preload_cache', action='store_true', help='Preload cache for dataset')

    args = parser.parse_args()

    main(args)
