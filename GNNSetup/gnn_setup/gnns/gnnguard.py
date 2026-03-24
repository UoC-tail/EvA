# This file implements GNNGuard as described in "GNNGuard: Defending Graph Neural Networks 
# against Adversarial Attacks" (NeurIPS 2020) adapted to match the code style of this project.

import collections
from typing import Callable, Dict, Optional, Sequence, Tuple, Union
from torchtyping import TensorType

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops

from torch_scatter import scatter_add
from torch_sparse import coalesce, SparseTensor

from gnn_setup.gnns.helpers.aggregation import chunked_message_and_aggregate
from gnn_setup.gnns.helpers.utils import (get_approx_topk_ppr_matrix, get_ppr_matrix, get_truncated_svd, get_jaccard,
                                        sparse_tensor_to_tuple, tuple_to_sparse_tensor)


class GNNGuardConv(GCNConv):
    """Extension of GCNConv for GNNGuard that allows chaining and includes attention mechanisms.

    Parameters
    ----------
    See https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#module-torch_geometric.nn.conv.gcn
    """

    def __init__(self, do_chunk: bool = False, n_chunks: int = 8, *input, **kwargs):
        super().__init__(*input, **kwargs)
        self.do_chunk = do_chunk
        self.n_chunks = n_chunks

    def forward(self, arguments: Tuple[TensorType["n_nodes", "n_features"],
                                       Union[TensorType[2, "nnz"], SparseTensor],
                                       Optional[TensorType["nnz"]]]) -> TensorType["n_nodes", "n_classes"]:
        """Predictions based on the input with GNNGuard attention mechanism.

        Parameters
        ----------
        arguments : Sequence[torch.Tensor]
            [x, edge indices] or [x, edge indices, edge weights], by default None

        Returns
        -------
        torch.Tensor
            the output of `GCNConv`.

        Raises
        ------
        NotImplementedError
            if the arguments are not of length 2 or 3
        """
        if len(arguments) == 2:
            x, edge_index = arguments
            edge_weight = None
        elif len(arguments) == 3:
            x, edge_index, edge_weight = arguments
        else:
            raise NotImplementedError("This method is just implemented for two or three arguments")
        
        embedding = super(GNNGuardConv, self).forward(x, edge_index, edge_weight=edge_weight)
        return embedding

    def message_and_aggregate(self, adj_t: Union[torch.Tensor, SparseTensor], x: torch.Tensor) -> torch.Tensor:
        if not self.do_chunk or not isinstance(adj_t, SparseTensor):
            return super(GNNGuardConv, self).message_and_aggregate(adj_t, x)
        else:
            return chunked_message_and_aggregate(adj_t, x, n_chunks=self.n_chunks)


ACTIVATIONS = {
    "ReLU": nn.ReLU(),
    "Tanh": nn.Tanh(),
    "ELU": nn.ELU(),
    "Identity": nn.Identity()
}


class GNNGuard(nn.Module):
    """GNNGuard implementation for defending against adversarial attacks on graphs.
    
    Based on "GNNGuard: Defending Graph Neural Networks against Adversarial Attacks" (NeurIPS 2020).
    This implementation includes attention mechanisms, edge pruning, and layer-wise graph memory.

    Parameters
    ----------
    n_features : int
        Number of attributes for each node
    n_classes : int
        Number of classes for prediction
    activation : Union[str, nn.Module], optional
        Arbitrary activation function for the hidden layer, by default nn.ReLU()
    n_filters : Union[int, Sequence[int]], optional
        Number of dimensions for the hidden units, by default 64
    bias : bool, optional
        If set to False, the layers will not learn an additive bias, by default True
    dropout : float, optional
        Dropout rate, by default 0.5
    with_batch_norm : bool, optional
        Whether to use batch normalization, by default False
    prune_edges : bool, optional
        Whether to enable learnable edge pruning, by default True
    mimic_ref_impl : bool, optional
        Whether to mimic reference implementation quirks, by default False
    div_limit : Union[str, float], optional
        Division limit for numerical stability, by default "auto"
    do_checkpoint : bool, optional
        If true use checkpointing in message passing, by default False
    n_chunks : int, optional
        Number of chunks for checkpointing, by default 8
    """

    def __init__(self,
                 n_features: int,
                 n_classes: int,
                 activation: Union[str, nn.Module] = nn.ReLU(),
                 n_filters: Union[int, Sequence[int]] = 64,
                 bias: bool = True,
                 dropout: float = 0.5,
                 with_batch_norm: bool = False,
                 prune_edges: bool = True,
                 mimic_ref_impl: bool = False,
                 div_limit: Union[str, float] = "auto",
                 do_checkpoint: bool = False,
                 n_chunks: int = 8,
                 **kwargs):
        super().__init__()
        
        if not isinstance(n_filters, Sequence):
            self.n_filters = [n_filters]
        else:
            self.n_filters = list(n_filters)
            
        if isinstance(activation, str):
            if activation in ACTIVATIONS.keys():
                self.activation = ACTIVATIONS[activation]
            else:
                raise AttributeError(f"Activation {activation} is not defined.")
        else:
            self.activation = activation

        self.n_features = n_features
        self.bias = bias
        self.n_classes = n_classes
        self.dropout = dropout
        self.with_batch_norm = with_batch_norm
        self.prune_edges = prune_edges
        self.mimic_ref_impl = mimic_ref_impl
        self.div_limit = div_limit
        self.do_checkpoint = do_checkpoint
        self.n_chunks = n_chunks
        
        # GNNGuard specific parameters
        self.pre_beta = nn.Parameter(torch.empty(()))
        self.pruning_weight = nn.Parameter(torch.empty(2)) if prune_edges else None
        
        self.layers = self._build_layers()
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters for GNNGuard."""
        # Initialize beta parameter for layer-wise graph memory
        nn.init.uniform_(self.pre_beta)
        
        # Initialize pruning weights if edge pruning is enabled
        if self.pruning_weight is not None:
            nn.init.xavier_uniform_(self.pruning_weight.unsqueeze(0))

    def _build_conv_layer(self, in_channels: int, out_channels: int):
        return GNNGuardConv(in_channels=in_channels, out_channels=out_channels,
                                     do_chunk=self.do_checkpoint, n_chunks=self.n_chunks, bias=self.bias)

    def _build_layers(self):
        filter_dimensions = [self.n_features] + self.n_filters
        modules = nn.ModuleList([
            nn.Sequential(collections.OrderedDict(
                [(f'gcn_{idx}', self._build_conv_layer(in_channels=in_channels, out_channels=out_channels))]
                + ([(f'bn_{idx}', torch.nn.BatchNorm1d(out_channels))] if self.with_batch_norm else [])
                + [(f'activation_{idx}', self.activation),
                   (f'dropout_{idx}', nn.Dropout(p=self.dropout))]
            ))
            for idx, (in_channels, out_channels)
            in enumerate(zip(filter_dimensions[:-1], self.n_filters))
        ])
        idx = len(modules)
        modules.append(nn.Sequential(collections.OrderedDict([
            (f'gcn_{idx}', self._build_conv_layer(in_channels=filter_dimensions[-1], out_channels=self.n_classes)),
        ])))
        return modules

    def forward(self,
                data: Optional[Union[Data, TensorType["n_nodes", "n_features"]]] = None,
                adj: Optional[Union[SparseTensor,
                                    torch.sparse.FloatTensor,
                                    Tuple[TensorType[2, "nnz"], TensorType["nnz"]],
                                    TensorType["n_nodes", "n_nodes"]]] = None,
                attr_idx: Optional[TensorType["n_nodes", "n_features"]] = None,
                edge_idx: Optional[TensorType[2, "nnz"]] = None,
                edge_weight: Optional[TensorType["nnz"]] = None,
                n: Optional[int] = None,
                d: Optional[int] = None) -> TensorType["n_nodes", "n_classes"]:
        
        x, edge_idx, edge_weight = self.parse_forward_input(data, adj, attr_idx, edge_idx, edge_weight, n, d)

        device = next(self.parameters()).device
        if x.device != device:
            x = x.to(device)
        if edge_idx.device != device:
            edge_idx = edge_idx.to(device)
        if edge_weight is not None and edge_weight.device != device:
            edge_weight = edge_weight.to(device)

        # Ensure contiguousness
        x, edge_idx, edge_weight = self._ensure_contiguousness(x, edge_idx, edge_weight)

        # Convert to dense adjacency matrix for attention computation
        n_nodes = x.shape[0]
        if isinstance(edge_idx, SparseTensor):
            adj_matrix = edge_idx.to_dense()
        else:
            adj_matrix = torch.sparse_coo_tensor(edge_idx, edge_weight, (n_nodes, n_nodes)).to_dense()

        # GNNGuard layer-wise processing with attention
        beta = torch.sigmoid(self.pre_beta)  # Ensure beta is between 0 and 1
        W = None  # Initialize graph memory
        
        for idx, layer in enumerate(self.layers):
            # Compute attention weights
            alpha = self._compute_attention_weights(adj_matrix, x)
            
            if idx == 0:
                W = alpha
            else:
                # Layer-wise graph memory: W = beta * W + (1 - beta) * alpha
                W = beta * W + (1 - beta) * alpha
            
            # Convert back to sparse format for GCN layer
            edge_idx_new, edge_weight_new = self._dense_to_sparse(W)
            
            # Apply GCN layer
            x = layer((x, edge_idx_new, edge_weight_new))

        return x

    def _compute_attention_weights(self, adj_matrix: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute attention weights based on cosine similarity and adjacency matrix.
        
        Parameters
        ----------
        adj_matrix : torch.Tensor
            Dense adjacency matrix [n_nodes, n_nodes]
        x : torch.Tensor
            Node features [n_nodes, n_features]
            
        Returns
        -------
        torch.Tensor
            Attention-weighted adjacency matrix
        """
        # Detach features to prevent gradients (as in original implementation)
        x_detached = x.detach()
        
        # Compute pairwise cosine similarity
        x_norm = F.normalize(x_detached, p=2, dim=1)
        cos_sim = torch.mm(x_norm, x_norm.t())
        
        if self.mimic_ref_impl:
            # Threshold cosine similarity (from reference implementation)
            cos_sim[cos_sim < 0.1] = 0
        
        # Multiply with adjacency matrix to get similarity-weighted edges
        S = adj_matrix * cos_sim
        
        # Normalize to get attention weights
        if not self.mimic_ref_impl:
            # Paper normalization
            N = (S != 0).float().sum(dim=-1)
            S_sums = S.sum(dim=-1)
            S_sums[S_sums.abs() < (1e-8 if self.div_limit == "auto" else self.div_limit)] = 1
            alpha = S * (N / ((N + 1) * S_sums)).unsqueeze(-1)
        else:
            # Reference implementation normalization
            S_sums = S.abs().sum(dim=-1)
            eps = 10 * torch.finfo(S.dtype).eps if self.div_limit == "auto" else self.div_limit
            S_sums[S_sums < eps] = 1
            alpha = S / S_sums.unsqueeze(-1)
        
        # Edge pruning
        if self.prune_edges and self.pruning_weight is not None:
            alpha = self._apply_edge_pruning(alpha)
        
        # Add self-loops
        if self.mimic_ref_impl:
            N = (alpha != 0).float().sum(dim=-1)
        else:
            N = (S != 0).float().sum(dim=-1)
        
        diag_values = 1.0 / (N + 1)
        alpha = alpha + torch.diag(diag_values)
        
        if self.mimic_ref_impl:
            # Apply exponential (from reference implementation)
            alpha = torch.exp(alpha)
            alpha[adj_matrix == 0] = 0  # Keep sparsity pattern
        
        return alpha

    def _apply_edge_pruning(self, alpha: torch.Tensor) -> torch.Tensor:
        """Apply learnable edge pruning to the attention matrix.
        
        Parameters
        ----------
        alpha : torch.Tensor
            Attention-weighted adjacency matrix
            
        Returns
        -------
        torch.Tensor
            Pruned attention matrix
        """
        # Get edge indices
        edges = alpha.nonzero()
        if edges.size(0) == 0:
            return alpha
        
        # Compute characteristic vector for each edge
        char_vec = torch.stack([
            alpha[edges[:, 0], edges[:, 1]], 
            alpha[edges[:, 1], edges[:, 0]]
        ])
        
        # Compute drop scores
        drop_score = torch.sigmoid(self.pruning_weight @ char_vec)
        
        # Prune edges with drop_score <= 0.5
        alpha_pruned = alpha.clone()
        mask = drop_score <= 0.5
        alpha_pruned[edges[mask, 0], edges[mask, 1]] = 0
        
        return alpha_pruned

    def _dense_to_sparse(self, dense_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert dense adjacency matrix to sparse format.
        
        Parameters
        ----------
        dense_adj : torch.Tensor
            Dense adjacency matrix
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Edge indices and edge weights
        """
        # Convert to sparse and get indices and values
        sparse_adj = dense_adj.to_sparse()
        edge_idx = sparse_adj.indices()
        edge_weight = sparse_adj.values()
        
        return edge_idx, edge_weight

    @staticmethod
    def parse_forward_input(data: Optional[Union[Data, TensorType["n_nodes", "n_features"]]] = None,
                            adj: Optional[Union[SparseTensor,
                                                torch.sparse.FloatTensor,
                                                Tuple[TensorType[2, "nnz"], TensorType["nnz"]],
                                                TensorType["n_nodes", "n_nodes"]]] = None,
                            attr_idx: Optional[TensorType["n_nodes", "n_features"]] = None,
                            edge_idx: Optional[TensorType[2, "nnz"]] = None,
                            edge_weight: Optional[TensorType["nnz"]] = None,
                            n: Optional[int] = None,
                            d: Optional[int] = None) -> Tuple[TensorType["n_nodes", "n_features"],
                                                              TensorType[2, "nnz"],
                                                              TensorType["nnz"]]:
        edge_weight = None
        # PyTorch Geometric support
        if isinstance(data, Data):
            x, edge_idx = data.x, data.edge_index
        # Randomized smoothing support
        elif attr_idx is not None and edge_idx is not None and n is not None and d is not None:
            x = coalesce(attr_idx, torch.ones_like(attr_idx[0], dtype=torch.float32), m=n, n=d)
            x = torch.sparse.FloatTensor(x[0], x[1], torch.Size([n, d])).to_dense()
            edge_idx = edge_idx
        # Empirical robustness support
        elif isinstance(adj, tuple):
            # Necessary since `torch.sparse.FloatTensor` eliminates the gradient...
            x, edge_idx, edge_weight = data, adj[0], adj[1]
        elif isinstance(adj, SparseTensor):
            x = data
            edge_idx_rows, edge_idx_cols, edge_weight = adj.coo()
            edge_idx = torch.stack([edge_idx_rows, edge_idx_cols], dim=0)
        else:
            if not adj.is_sparse:
                adj = adj.to_sparse()
            x, edge_idx, edge_weight = data, adj._indices(), adj._values()

        if edge_weight is None:
            edge_weight = torch.ones_like(edge_idx[0], dtype=torch.float32)

        if edge_weight.dtype != torch.float32:
            edge_weight = edge_weight.float()

        return x, edge_idx, edge_weight

    def _ensure_contiguousness(self,
                               x: torch.Tensor,
                               edge_idx: Union[torch.Tensor, SparseTensor],
                               edge_weight: Optional[torch.Tensor]) -> Tuple[TensorType["n_nodes", "n_features"],
                                                                             Union[TensorType[2, "nnz"], SparseTensor],
                                                                             Optional[TensorType["nnz"]]]:
        if not x.is_sparse:
            x = x.contiguous()
        if hasattr(edge_idx, 'contiguous'):
            edge_idx = edge_idx.contiguous()
        if edge_weight is not None:
            edge_weight = edge_weight.contiguous()
        return x, edge_idx, edge_weight
