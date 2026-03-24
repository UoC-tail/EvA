import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_sparse import SparseTensor
from gnn_setup.gnns.gcn import GCN


class GCNSVD(GCN):
    """GCN with Truncated SVD preprocessing for defense against adversarial attacks.
    
    Based on "All You Need Is Low (Rank): Defending Against Adversarial Attacks on Graphs"
    https://dl.acm.org/doi/abs/10.1145/3336191.3371789
    
    This implementation extends the base GCN class and applies SVD preprocessing
    by setting the appropriate svd_params.
    
    Parameters
    ----------
    n_features : int
        Number of input features
    n_hidden : int or n_filters
        Number of hidden units (mapped to n_filters internally)
    n_classes : int
        Number of output classes
    dropout : float, optional
        Dropout rate (default: 0.5)
    k : int, optional
        Rank for SVD truncation (default: 50)
    **kwargs
        Additional arguments passed to the base GCN class
    """
    
    def __init__(self, n_features, n_hidden=64, n_classes=2, dropout=0.5, k=50, **kwargs):
        # Map n_hidden to n_filters for compatibility with base GCN
        if 'n_filters' not in kwargs:
            kwargs['n_filters'] = n_hidden
            
        # Set SVD parameters
        svd_params = {'rank': k}
        
        # Initialize base GCN with SVD preprocessing
        super(GCNSVD, self).__init__(
            n_features=n_features,
            n_classes=n_classes,
            dropout=dropout,
            svd_params=svd_params,
            **kwargs
        )
        
        self.k = k
        self.n_hidden = n_hidden
