import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_sparse import SparseTensor
from tqdm import tqdm
from numba import njit
from gnn_setup.gnns.gcn import GCN


class GCNJaccard(GCN):
    """GCN with Jaccard similarity preprocessing for defense against adversarial attacks.
    
    Based on "Adversarial Examples on Graph Data: Deep Insights into Attack and Defense"
    https://arxiv.org/pdf/1903.01610.pdf
    
    This implementation extends the base GCN class and applies Jaccard preprocessing
    by setting the appropriate jaccard_params.
    
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
    threshold : float, optional
        Similarity threshold for dropping edges (default: 0.01)
    binary_feature : bool, optional
        Whether features are binary (default: True)
    **kwargs
        Additional arguments passed to the base GCN class
    """
    
    def __init__(self, n_features, n_hidden=64, n_classes=2, dropout=0.5, 
                 threshold=0.01, binary_feature=False, **kwargs):
        # Map n_hidden to n_filters for compatibility with base GCN
        if 'n_filters' not in kwargs:
            kwargs['n_filters'] = n_hidden
            
        # Set Jaccard parameters
        jaccard_params = {
            'threshold': threshold,
        }
        
        # Initialize base GCN with Jaccard preprocessing
        super(GCNJaccard, self).__init__(
            n_features=n_features,
            n_classes=n_classes,
            dropout=dropout,
            jaccard_params=jaccard_params,
            **kwargs
        )
        
        self.threshold = threshold
        self.binary_feature = binary_feature
        self.n_hidden = n_hidden
