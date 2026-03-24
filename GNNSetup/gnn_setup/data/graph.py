# This file is taken from saper0/adversarial_training on Github
# This file has been mostly taken from the work bei Geisler et al. 
# "Robustness of Graph Neural Networks at Scale" (NeurIPS, 2021) and adapted
# for this work: https://github.com/sigeisler/robustness_of_gnns_at_scale

import logging
import warnings
from typing import Any, Dict, Iterable, List, Union, Tuple, Optional

import numpy as np
import pandas as pd
from ogb.nodeproppred import PygNodePropPredDataset
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

import torch

sparse_graph_properties = [
    'adj_matrix', 'attr_matrix', 'labels', 'node_names', 'attr_names', 'class_names', 'metadata'
]

class SparseGraph:
    """Attributed labeled graph stored in sparse matrix form.

    All properties are immutable so users don't mess up the
    data format's assumptions.
    Be careful when circumventing this and changing the internal matrices
    regardless (e.g. by exchanging the data array of a sparse matrix).

    Parameters
    ----------
    adj_matrix
        Adjacency matrix in CSR format. Shape [num_nodes, num_nodes]
    attr_matrix
        Attribute matrix in CSR or numpy format. Shape [num_nodes, num_attr]
    labels
        Array, where each entry represents respective node's label(s). Shape [num_nodes]
        Alternatively, CSR matrix with labels in one-hot format. Shape [num_nodes, num_classes]
    node_names
        Names of nodes (as strings). Shape [num_nodes]
    attr_names
        Names of the attributes (as strings). Shape [num_attr]
    class_names
        Names of the class labels (as strings). Shape [num_classes]
    metadata
        Additional metadata such as text.

    """

    def __init__(
            self, adj_matrix: sp.spmatrix,
            attr_matrix: Union[np.ndarray, sp.spmatrix] = None,
            labels: Union[np.ndarray, sp.spmatrix] = None,
            node_names: np.ndarray = None,
            attr_names: np.ndarray = None,
            class_names: np.ndarray = None,
            metadata: Any = None):
        # Make sure that the dimensions of matrices / arrays all agree
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr().astype(np.float32)
        else:
            raise ValueError("Adjacency matrix must be in sparse format (got {0} instead)."
                             .format(type(adj_matrix)))

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree.")

        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError("Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)."
                                 .format(type(attr_matrix)))

            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency and attribute matrices don't agree.")

        if labels is not None:
            if labels.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the label vector don't agree.")

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the node names don't agree.")

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError("Dimensions of the attribute matrix and the attribute names don't agree.")

        self.adj_matrix = adj_matrix
        self.attr_matrix = attr_matrix
        self.labels = labels
        self.node_names = node_names
        self.attr_names = attr_names
        self.class_names = class_names
        self.metadata = metadata

    def num_nodes(self) -> int:
        """Get the number of nodes in the graph.
        """
        return self.adj_matrix.shape[0]

    def num_edges(self, warn: bool = True) -> int:
        """Get the number of edges in the graph.

        For undirected graphs, (i, j) and (j, i) are counted as _two_ edges.

        """
        if warn and not self.is_directed():
            warnings.warn("num_edges always returns the number of directed edges now.", FutureWarning)
        return self.adj_matrix.nnz

    def get_neighbors(self, idx: int) -> np.ndarray:
        """Get the indices of neighbors of a given node.

        Parameters
        ----------
        idx
            Index of the node whose neighbors are of interest.

        """
        return self.adj_matrix[idx].indices

    def get_edgeid_to_idx_array(self) -> np.ndarray:
        """Return a Numpy Array that maps edgeids to the indices in the adjacency matrix.

        Returns
        -------
        np.ndarray
            The i'th entry contains the x- and y-coordinates of edge i in the adjacency matrix.
            Shape [num_edges, 2]

        """
        return np.transpose(self.adj_matrix.nonzero())

    def get_idx_to_edgeid_matrix(self) -> sp.csr_matrix:
        """Return a sparse matrix that maps indices in the adjacency matrix to edgeids.

        Caution: This contains one explicit 0 (zero stored as a nonzero),
        which is the index of the first edge.

        Returns
        -------
        sp.csr_matrix
            The entry [x, y] contains the edgeid of the corresponding edge (or 0 for non-edges).
            Shape [num_nodes, num_nodes]

        """
        return sp.csr_matrix(
            (np.arange(self.adj_matrix.nnz), self.adj_matrix.indices, self.adj_matrix.indptr),
            shape=self.adj_matrix.shape)

    def is_directed(self) -> bool:
        """Check if the graph is directed (adjacency matrix is not symmetric).
        """
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0

    def to_undirected(self) -> 'SparseGraph':
        """Convert to an undirected graph (make adjacency matrix symmetric).
        """
        idx = self.get_edgeid_to_idx_array().T
        ridx = np.ravel_multi_index(idx, self.adj_matrix.shape)
        ridx_rev = np.ravel_multi_index(idx[::-1], self.adj_matrix.shape)

        # Get duplicate edges (self-loops and opposing edges)
        dup_ridx = ridx[np.isin(ridx, ridx_rev)]
        dup_idx = np.unravel_index(dup_ridx, self.adj_matrix.shape)

        # Check if the adjacency matrix weights are symmetric (if nonzero)
        if len(dup_ridx) > 0 and not np.allclose(self.adj_matrix[dup_idx], self.adj_matrix[dup_idx[::-1]]):
            raise ValueError("Adjacency matrix weights of opposing edges differ.")

        # Create symmetric matrix
        new_adj_matrix = self.adj_matrix + self.adj_matrix.T
        if len(dup_ridx) > 0:
            new_adj_matrix[dup_idx] = (new_adj_matrix[dup_idx] - self.adj_matrix[dup_idx]).A1

        self.adj_matrix = new_adj_matrix
        return self

    def is_weighted(self) -> bool:
        """Check if the graph is weighted (edge weights other than 1).
        """
        return np.any(np.unique(self.adj_matrix[self.adj_matrix.nonzero()].A1) != 1)

    def to_unweighted(self) -> 'SparseGraph':
        """Convert to an unweighted graph (set all edge weights to 1).
        """
        self.adj_matrix.data = np.ones_like(self.adj_matrix.data)
        return self

    def is_connected(self) -> bool:
        """Check if the graph is connected.
        """
        return sp.csgraph.connected_components(self.adj_matrix, return_labels=False) == 1

    def has_self_loops(self) -> bool:
        """Check if the graph has self-loops.
        """
        return not np.allclose(self.adj_matrix.diagonal(), 0)

    def __repr__(self) -> str:

        dir_string = 'Directed' if self.is_directed() else 'Undirected'
        weight_string = 'weighted' if self.is_weighted() else 'unweighted'
        conn_string = 'connected' if self.is_connected() else 'disconnected'
        loop_string = 'has self-loops' if self.has_self_loops() else 'no self-loops'
        return ("<{}, {} and {} SparseGraph with {} edges ({})>"
                .format(dir_string, weight_string, conn_string,
                        self.num_edges(warn=False), loop_string))

    def _adopt_graph(self, graph: 'SparseGraph'):
        """Copy all properties from the given graph to this graph.
        """
        for prop in sparse_graph_properties:
            setattr(self, '_{}'.format(prop), getattr(graph, prop))

    # Quality of life (shortcuts)
    def standardize(
            self, make_unweighted: bool = True,
            make_undirected: bool = True,
            no_self_loops: bool = True,
            select_lcc: bool = True
    ) -> 'SparseGraph':
        """Perform common preprocessing steps: remove self-loops, make unweighted/undirected, select LCC.

        All changes are done inplace.

        Parameters
        ----------
        make_unweighted
            Whether to set all edge weights to 1.
        make_undirected
            Whether to make the adjacency matrix symmetric. Can only be used if make_unweighted is True.
        no_self_loops
            Whether to remove self loops.
        select_lcc
            Whether to select the largest connected component of the graph.

        """
        G = self
        if make_unweighted and G.is_weighted():
            G = G.to_unweighted()
        if make_undirected and G.is_directed():
            G = G.to_undirected()
        if no_self_loops and G.has_self_loops():
            G = remove_self_loops(G)
        if select_lcc and not G.is_connected():
            G = largest_connected_components(G, 1, make_undirected)
        self._adopt_graph(G)
        return G

    @staticmethod
    def from_flat_dict(data_dict: Dict[str, Any]) -> 'SparseGraph':
        """Initialize SparseGraph from a flat dictionary.
        """
        init_dict = {}
        del_entries = []

        # Construct sparse matrices
        for key in data_dict.keys():
            if key.endswith('_data') or key.endswith('.data'):
                if key.endswith('_data'):
                    sep = '_'
                else:
                    sep = '.'
                matrix_name = key[:-5]
                mat_data = key
                mat_indices = '{}{}indices'.format(matrix_name, sep)
                mat_indptr = '{}{}indptr'.format(matrix_name, sep)
                mat_shape = '{}{}shape'.format(matrix_name, sep)
                if matrix_name == 'adj' or matrix_name == 'attr':
                    matrix_name += '_matrix'
                init_dict[matrix_name] = sp.csr_matrix(
                    (data_dict[mat_data],
                     data_dict[mat_indices],
                     data_dict[mat_indptr]),
                    shape=data_dict[mat_shape])
                del_entries.extend([mat_data, mat_indices, mat_indptr, mat_shape])

        # Delete sparse matrix entries
        for del_entry in del_entries:
            del data_dict[del_entry]

        # Load everything else
        for key, val in data_dict.items():
            if ((val is not None) and (None not in val)):
                init_dict[key] = val

        # CHANGED: This block removes any unnecessary keys from the dictionary
        needed_keys = ["adj_matrix", "attr_matrix", "labels", "node_names", "attr_names", "class_names", "metadata"]
        
        refined_dict = dict()
        for key in init_dict.keys():
            if key in needed_keys:
                refined_dict[key] = init_dict[key]
            else:
                # remove the key from the dictionary
                pass
        init_dict = refined_dict

        print("I am here, ", list(init_dict.keys()))
        return SparseGraph(**init_dict)


def remove_self_loops(sparse_graph: SparseGraph) -> SparseGraph:
    """Remove self loops (diagonal entries in the adjacency matrix).

    Changes are returned in a partially new SparseGraph.

    """
    num_self_loops = (~np.isclose(sparse_graph.adj_matrix.diagonal(), 0)).sum()
    if num_self_loops > 0:
        adj_matrix = sparse_graph.adj_matrix.copy().tolil()
        adj_matrix.setdiag(0)
        adj_matrix = adj_matrix.tocsr()
        return SparseGraph(
            adj_matrix, sparse_graph.attr_matrix, sparse_graph.labels, sparse_graph.node_names,
            sparse_graph.attr_names, sparse_graph.class_names, sparse_graph.metadata)
    else:
        return sparse_graph
    

def largest_connected_components(sparse_graph: SparseGraph, n_components: int = 1, make_undirected=True) -> SparseGraph:
    """Select the largest connected components in the graph.

    Changes are returned in a partially new SparseGraph.

    Parameters
    ----------
    sparse_graph
        Input graph.
    n_components
        Number of largest connected components to keep.

    Returns
    -------
    SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = sp.csgraph.connected_components(sparse_graph.adj_matrix, directed=make_undirected)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    return create_subgraph(sparse_graph, nodes_to_keep=nodes_to_keep)

def create_subgraph(
        sparse_graph: SparseGraph,
        _sentinel: None = None,
        nodes_to_remove: np.ndarray = None,
        nodes_to_keep: np.ndarray = None
) -> SparseGraph:
    """Create a graph with the specified subset of nodes.

    Exactly one of (nodes_to_remove, nodes_to_keep) should be provided, while the other stays None.
    Note that to avoid confusion, it is required to pass node indices as named arguments to this function.

    The subgraph partially points to the old graph's data.

    Parameters
    ----------
    sparse_graph
        Input graph.
    _sentinel
        Internal, to prevent passing positional arguments. Do not use.
    nodes_to_remove
        Indices of nodes that have to removed.
    nodes_to_keep
        Indices of nodes that have to be kept.

    Returns
    -------
    SparseGraph
        Graph with specified nodes removed.

    """
    # Check that arguments are passed correctly
    if _sentinel is not None:
        raise ValueError("Only call `create_subgraph` with named arguments',"
                         " (nodes_to_remove=...) or (nodes_to_keep=...).")
    if nodes_to_remove is None and nodes_to_keep is None:
        raise ValueError("Either nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None and nodes_to_keep is not None:
        raise ValueError("Only one of nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None:
        nodes_to_keep = [i for i in range(sparse_graph.num_nodes()) if i not in nodes_to_remove]
    elif nodes_to_keep is not None:
        nodes_to_keep = sorted(nodes_to_keep)
    else:
        raise RuntimeError("This should never happen.")

    adj_matrix = sparse_graph.adj_matrix[nodes_to_keep][:, nodes_to_keep]
    if sparse_graph.attr_matrix is None:
        attr_matrix = None
    else:
        attr_matrix = sparse_graph.attr_matrix[nodes_to_keep]
    if sparse_graph.labels is None:
        labels = None
    else:
        labels = sparse_graph.labels[nodes_to_keep]
    if sparse_graph.node_names is None:
        node_names = None
    else:
        node_names = sparse_graph.node_names[nodes_to_keep]
    return SparseGraph(
        adj_matrix, attr_matrix, labels, node_names, sparse_graph.attr_names,
        sparse_graph.class_names, sparse_graph.metadata)


def receptive_field_subgraph(attr, adj, labels=None, target_mask=None, n_hops=2):
    if target_mask is None:
        target_mask = torch.ones(size=(attr.shape[0], ), dtype=torch.bool())
        in_target = torch.tensor(target_mask).float().to(attr.device)
    elif (isinstance(target_mask, torch.Tensor) or isinstance(target_mask, np.ndarray)) and target_mask.dtype != torch.bool:
        in_target = torch.zeros(size=(attr.shape[0], ), dtype=torch.float32).to(attr.device)
        in_target[target_mask] = 1
    out_target = in_target.clone()
    for _ in range(n_hops):
        out_target += torch.sparse.mm(adj, out_target.unsqueeze(1)).squeeze(1)
        out_target = (out_target > 0).float()
    filter_mask = out_target.bool()
    attr_filtered = attr[filter_mask]

    # creating adj filtered
    filter_idx_map = filter_mask.long().cumsum(dim=0) - 1
    filter_idx_map[~filter_mask] = -1

    adj_filter_row = filter_mask[adj.indices()[0]]
    adj_filter_cols = filter_mask[adj.indices()[1]]
    filter_mask_adj = adj_filter_row & adj_filter_cols

    adj_filtered = torch.sparse_coo_tensor(
        indices=torch.stack([
            filter_idx_map[adj.indices()[0]][filter_mask_adj], 
            filter_idx_map[adj.indices()[1]][filter_mask_adj]]),
        values=adj.values()[filter_mask_adj],
        size=(attr_filtered.shape[0], attr_filtered.shape[0])
    ).coalesce()
    
    if labels is not None:
        labels_filtered = labels[filter_mask]
        return attr_filtered, adj_filtered, labels_filtered, filter_idx_map, filter_mask
    else:
        return attr_filtered, adj_filtered, filter_idx_map, filter_mask
    

def return_subgraph_to_original(attr_reduced, attr_original, 
                                adj_reduced, adj_original, 
                                filter_idx_map, filter_mask,
                                labels_reduced=None, labels_original=None):
    attr_reconst = attr_original.clone()
    attr_reconst[filter_mask] = attr_reduced
    
    # computing the reverse filter map
    reverse_filter_map = torch.arange(0, filter_idx_map.shape[0]).to(filter_mask.device)
    reverse_filter_map = reverse_filter_map[filter_mask]

    # reconstructing the adjacency matrix
    reduced_indices = adj_reduced.coalesce().indices()
    reconst_indices = reverse_filter_map[reduced_indices]
    internal_adj = torch.sparse_coo_tensor(
        indices=reconst_indices,
        values=adj_reduced.coalesce().values(),
        size=adj_original.size()
    ).coalesce()

    original_indices = adj_original.coalesce().indices()
    
    remaining_filter = ((~filter_mask[original_indices]).sum(dim=0)) > 0
    remaining_mat_indices = original_indices[:, remaining_filter]
    remaining_adj = torch.sparse_coo_tensor(
        indices=remaining_mat_indices,
        values=adj_original.coalesce().values()[remaining_filter],
        size=adj_original.size()
    ).coalesce()
    adj_reconst = (internal_adj + remaining_adj).coalesce()
    
    if labels_reduced is not None:
        labels_reconst = labels_original.clone()
        labels_reconst[filter_mask] = labels_reduced
        return attr_reconst, adj_reconst, labels_reconst
    else:
        return attr_reconst, adj_reconst