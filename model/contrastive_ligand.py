import os
import shutil
import torch
import yaml
import numpy as np
from datetime import datetime

import torch.nn as nn
import torch.nn.functional as F

from gnn import GCN, GINet

class MolCLRLigand(nn.Module):
    """
    Contrastive learning framework for protein-ligand interactions
    Adapted from MolCLR for ligand binding prediction
    """

    def __init__(self, 
                 model_name="gcn", 
                 num_layer=5, 
                 emb_dim=300, 
                 feat_dim=256, 
                 drop_ratio=0,
                 device='cuda',
                 pool='mean'):
        super(MolCLRLigand, self).__init__()
        self.model_name = model_name
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        self.pool = pool
        if model_name.lower() == 'gcn':
            self.model = GCN(num_layer=num_layer, emb_dim=emb_dim, feat_dim=feat_dim, drop_ratio=drop_ratio, pool=pool) 
        elif model_name.lower() == 'gin':
            self.model = GINet(num_layer=num_layer, emb_dim=emb_dim, feat_dim=feat_dim, drop_ratio=drop_ratio, pool=pool)
        else:
            raise ValueError(f"Unsupported model type: {model_name}. Supported types: 'gcn', 'gin'.")
        
        self.to(device)

    def forward(self, data_i, data_j):
        # get the representations and the projections
        ris, zis = self.model(data_i)  # [N,C]

        # get the representations and the projections
        rjs, zjs = self.model(data_j)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        return zis, zjs
