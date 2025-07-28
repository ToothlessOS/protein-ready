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
from pytorch_lightning.strategies import DDPStrategy
import torch.multiprocessing as mp

from model import MInterface, MInterfaceLigand
from data import DInterface, DInterfaceLigand
from utils import load_model_path_by_args

# Set multiprocessing start method for CUDA compatibility
try:
    mp.set_start_method('spawn', force=True)
    print("Multiprocessing: Set start method to 'spawn' for CUDA compatibility")
except RuntimeError as e:
    print(f"Multiprocessing: Could not set start method - {e}")

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
        devices=getattr(args, 'devices', 'cuda'),
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
        strategy=DDPStrategy(find_unused_parameters=True)
    )
    
    ckpt_path = getattr(args, 'ckpt_path', None)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)

# Pretrain ligand model
def ligand(args):
    pl.seed_everything(args.seed)
    
    # Set ligand-specific defaults if not already set
    if not hasattr(args, 'dataset') or args.dataset == 'protein_dataset':
        args.dataset = 'ligand_dataset'
    if not hasattr(args, 'model_name') or args.model_name == 'contrastive':
        args.model_name = 'gcn'  # Default to GCN for ligand models
    if not hasattr(args, 'loss') or args.loss == 'contrastive':
        args.loss = 'nt_xent'  # Use NT-Xent loss for ligand contrastive learning
    if not hasattr(args, 'data_path') or args.data_path == 'dataset/protein_g/':
        args.data_path = 'dataset/MolCLR_data/data/pubchem-10m-clean.txt'  # Default ligand data path
    
    # Load model path for resuming training if specified
    load_path = load_model_path_by_args(args)
    
    # Initialize ligand data module
    data_module = DInterfaceLigand(**vars(args))
    
    # Initialize ligand model
    if load_path is None:
        model = MInterfaceLigand(**vars(args))
    else:
        model = MInterfaceLigand(**vars(args))
        args.ckpt_path = load_path
    
    # Load callbacks (same as main function)
    callbacks = load_callbacks()
    
    # Initialize trainer with same configuration as main function
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
        strategy=DDPStrategy(find_unused_parameters=True)
    )
    
    # Start training
    ckpt_path = getattr(args, 'ckpt_path', None)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)

# Downstream tasks

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
    parser.add_argument('--lr_decay_min_lr', default=1e-6, type=float)
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
    parser.add_argument('--pdb_path', default='dataset/rcsb/human/', type=str, help='Path to PDB files for on-the-fly processing')
    parser.add_argument('--cache_path', default='dataset/protein_cache/', type=str, help='Path for persistent cache storage')
    parser.add_argument('--enable_pdb_processing', action='store_true', default=True, help='Enable on-the-fly PDB processing')
    parser.add_argument('--force_pdb_processing', action='store_true', help='Force PDB processing even if .pt files exist')
    parser.add_argument('--skip_invalid_files', action='store_true', default=True, help='Skip files that fail to process')
    parser.add_argument('--validate_on_init', action='store_true', help='Pre-validate PDB files during dataset initialization')
    parser.add_argument('--esm_use_cpu', action='store_true', help='Force ESM-C to use CPU (avoids CUDA multiprocessing issues)')
    parser.add_argument('--complete_graph_percent', default=0, type=int, help='Percentage of complete graphs in the dataset') # Different from GearNet
    parser.add_argument('--test_percent', default=0.01, type=float, help='Percentage of data used for testing')
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
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int, help='Run validation every N epochs')
    parser.add_argument('--val_check_interval', default=1.0, type=float, help='Validation check interval (fraction of epoch or steps)')
    parser.add_argument('--num_sanity_val_steps', default=2, type=int, help='Number of validation steps to run before training')
    parser.add_argument('--log_every_n_steps', default=1, type=int, help='Log every N steps (set to 1 for small datasets)')
    
    # Dataset and caching parameters
    parser.add_argument('--cache_size', default=512, type=int, help='Cache size for dataset')
    parser.add_argument('--enable_cache', action='store_true', help='Enable dataset caching')
    parser.add_argument('--preload_cache', action='store_true', help='Preload cache for dataset')
    
    # Ligand-specific parameters
    parser.add_argument('--mode', default='protein', choices=['protein', 'ligand'], type=str, 
                        help='Training mode: protein for protein pretraining, ligand for ligand pretraining')
    parser.add_argument('--valid_size', default=0.1, type=float, help='Validation set size for ligand training')
    parser.add_argument('--use_cosine_similarity', action='store_true', help='Use cosine similarity in NT-Xent loss')
    
    # Ligand model-specific parameters (for GCN/GIN)
    parser.add_argument('--num_layer', default=5, type=int, help='Number of GNN layers for ligand models')
    parser.add_argument('--emb_dim', default=300, type=int, help='Embedding dimension for ligand models')
    parser.add_argument('--feat_dim', default=256, type=int, help='Feature dimension for ligand models')
    parser.add_argument('--drop_ratio', default=0.0, type=float, help='Dropout ratio for ligand models')
    parser.add_argument('--pool', default='mean', choices=['mean', 'max', 'add'], help='Pooling method for ligand models')

    args = parser.parse_args()

    # Route to appropriate training function based on mode
    if args.mode == 'ligand':
        ligand(args)
    else:
        main(args)
