import os
from torch_geometric.datasets import CitationFull, Planetoid, Amazon
from ml_collections import ConfigDict
from torch_geometric.utils import to_scipy_sparse_matrix
import json
import torch
import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.propagate = True
logger.setLevel(logging.DEBUG)

from gnn_setup.data.graph import SparseGraph, largest_connected_components
from gnn_setup.data.split import Split
from gnn_setup.utils.storage import TensorHash
from gnn_setup.utils.namings import SplitNaming
from gnn_setup.utils.tensors import sparse_tensor
from gnn_setup.data.split import node_induced_subgraph, node_induced_subgraph_arxiv

import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset



def load_dataset(dataset_name, dataset_root, device="cpu"):
    # Load the dataset from the given dataset_root
    if dataset_name in ["cora", "cora_ml", "pubmed", "citeseer", 'amazon_computers', 'amazon_photo']:
        if dataset_name in ["cora_ml", "pubmed", "citeseer"]:
            dataset_obj = CitationFull(
                root=dataset_root, name=dataset_name)
        elif dataset_name == "amazon_computers":
            dataset_obj = Amazon(root=dataset_root, name='Computers')
        elif dataset_name == "amazon_photo":
            dataset_obj = Amazon(root=dataset_root, name='Photo')
        else:
            dataset_obj = Planetoid(
                root=dataset_root, name=dataset_name)
        dataset = dataset_obj[0].to(device)
        dataset_info = ConfigDict()
        dataset_info.n_features = dataset_obj.num_features
        dataset_info.n_classes = dataset_obj.num_classes
        dataset_info.n_nodes = dataset_obj.data.x.shape[0]
        dataset_info.dataset_name = dataset_name

        # dataset preprocessing with the largest connected component
        attr_matrix = dataset.x.cpu().numpy()
        adj_matrix = to_scipy_sparse_matrix(dataset.edge_index, num_nodes=dataset.x.shape[0])
        labels = dataset.y
        graphs = SparseGraph(adj_matrix, attr_matrix, labels)

        prep_graphs = largest_connected_components(sparse_graph=graphs, n_components=1, make_undirected=True)

        dataset.x = torch.tensor(prep_graphs.attr_matrix)
        dataset.edge_index = torch.tensor(prep_graphs.get_edgeid_to_idx_array().T, dtype=torch.long)
        dataset.y = torch.tensor(prep_graphs.labels)

        dataset_info.n_nodes = dataset.x.shape[0]
        return dataset, dataset_info
    else:
        raise NotImplementedError(
            f"Load method {dataset_name} not implemented yet.")

def load_dataset_split(dataset_name, split_name, dataset_root, splits_root, device="cpu"):
    split_file = os.path.join(splits_root, split_name)
    split_config_name = split_name.replace(".pt", "-conf.json")
    split_config_file = os.path.join(splits_root, split_config_name)
    try:
        split_dict = torch.load(split_file)
        split_config = json.load(open(split_config_file))
    
    except FileNotFoundError as e:
        raise RuntimeError(f"Error loading split file: {e}")

    dataset, dataset_info = load_dataset(dataset_name, dataset_root, device=device)
    training_idx = split_dict.get("training")
    validation_idx = split_dict.get("validation")
    test_idx = split_dict.get("test")
    unlabeled_idx = split_dict.get("unlabeled")

    check_split_valid(
        training_idx=training_idx, validation_idx=validation_idx, test_idx=test_idx,
        unlabeled_idx=unlabeled_idx, dataset_info=dataset_info, split_name=split_name, split_config=split_config)

    return {
        "training_idx": training_idx, 
        "validation_idx": validation_idx, 
        "test_idx": test_idx, 
        "unlabeled_idx": unlabeled_idx,
        "dataset_info": dataset_info, 
        "split_name": split_name,
        "config": split_config
    }

def make_dataset_split(dataset_name, 
                        training_nodes=None, validation_nodes=None, test_nodes=None, 
                        test_split_type=None, training_split_type=None, validation_split_type=None,
                        inductive=False, 
                        dataset_root="data", splits_root="splits", device="cpu", save=True):
    
    # region Loading the dataset, and creating splits
    dataset, dataset_info = load_dataset(dataset_name, dataset_root)
    
    split = Split(dataset)
    training_mask = split.alloc(
        budget=training_nodes,
        budget_type='per_class' if training_nodes > 1 else 'overall',
        stratified=(training_split_type == 'stratified'))
    training_idx = training_mask.nonzero(as_tuple=True)[0]
    validation_mask = split.alloc(
        budget=validation_nodes,
        budget_type='per_class' if validation_nodes > 1 else 'overall',
        stratified=(validation_split_type == 'stratified'))
    validation_idx = validation_mask.nonzero(as_tuple=True)[0]

    test_mask = split.alloc(
        budget=test_nodes,
        budget_type='per_class' if test_nodes > 1 else 'overall',
        stratified=(test_split_type == 'stratified'))
    test_idx = test_mask.nonzero(as_tuple=True)[0]

    unlabeled_mask = ~(training_mask | validation_mask | test_mask)
    unlabeled_idx = unlabeled_mask.nonzero(as_tuple=True)[0]
    # endregion

    # region Storing the splits
    split_dict = {
        "training": training_idx.cpu(), 
        "validation": validation_idx.cpu(),
        "unlabeled": unlabeled_idx.cpu(),
        "test": test_idx.cpu(),
        "config": {
            "dataset_name": dataset_name,
            "training_nodes": training_nodes,
            "validation_nodes": validation_nodes,
            "test_nodes": test_nodes,
            "test_split_type": test_split_type,
            "training_split_type": training_split_type,
            "validation_split_type": validation_split_type,
        }}
    split_name = TensorHash.hash_tensor_dict(split_dict)
    try:
        if save:
            os.makedirs(splits_root, exist_ok=True)
            torch.save(split_dict, os.path.join(splits_root, SplitNaming().name(dataset_name, split_name)))
            json.dump(split_dict.get("config"), open(os.path.join(splits_root, SplitNaming().conf_name(dataset_name, split_name)), "w"))
    except RuntimeError as e:
        print(f"Error saving the split: {e}")
    # endregion

    return {
        "training_idx": training_idx, 
        "validation_idx": validation_idx, 
        "test_idx": test_idx, 
        "unlabeled_idx": unlabeled_idx,
        "dataset_info": dataset_info, 
        "split_name": split_name,
        "config": split_dict.get("config", dict())
    }


def check_split_valid(
        training_idx, validation_idx, test_idx, 
        unlabeled_idx, dataset_info, split_name, split_config):
    assert torch.isin(training_idx, validation_idx).sum() == 0
    assert torch.isin(training_idx, test_idx).sum() == 0
    assert torch.isin(training_idx, unlabeled_idx).sum() == 0
    assert torch.isin(validation_idx, test_idx).sum() == 0
    assert torch.isin(validation_idx, unlabeled_idx).sum() == 0
    assert torch.isin(test_idx, unlabeled_idx).sum() == 0
    concat_idx = torch.cat([training_idx, validation_idx, test_idx, unlabeled_idx]).unique()
    assert concat_idx.shape[0] == dataset_info.n_nodes
    assert concat_idx.min() == 0
    assert concat_idx.max() == dataset_info.n_nodes - 1

    return True
    
def load_attr_adj(dataset, idx, device):
    out_graph = SparseGraph(
        adj_matrix=to_scipy_sparse_matrix(dataset.edge_index.cpu(), num_nodes=dataset.x.shape[0]),
        attr_matrix=dataset.x.cpu().numpy(),
        labels=dataset.y.cpu().numpy()
    )
    attr = torch.tensor(out_graph.attr_matrix).to(device)

    adj = sparse_tensor(out_graph.adj_matrix.tocoo()).to(device)
    return attr, adj

def splited_datasets(dataset, dataset_info, training_idx, validation_idx, unlabeled_idx, test_idx, inductive=False, return_idx=False):
    # converting idxs to masks
    training_mask = torch.zeros(dataset_info.n_nodes, dtype=torch.bool)
    training_mask[training_idx] = True
    validation_mask = torch.zeros(dataset_info.n_nodes, dtype=torch.bool)
    validation_mask[validation_idx] = True
    unlabeled_mask = torch.zeros(dataset_info.n_nodes, dtype=torch.bool)
    unlabeled_mask[unlabeled_idx] = True
    test_mask = torch.zeros(dataset_info.n_nodes, dtype=torch.bool)
    test_mask[test_idx] = True

    if inductive:
        training_dataset = node_induced_subgraph(dataset, training_mask | unlabeled_mask)
        validation_dataset = node_induced_subgraph(dataset, training_mask | validation_mask | unlabeled_mask)
        test_dataset = dataset
        inductive_trainig_mask = training_mask[training_mask | unlabeled_mask]
        inductive_validation_mask = validation_mask[training_mask | validation_mask | unlabeled_mask]
        refined_training_idx = inductive_trainig_mask.nonzero(as_tuple=True)[0]
        refined_validation_idx = inductive_validation_mask.nonzero(as_tuple=True)[0]
        refined_unlabeled_idx = (~training_mask[training_mask | unlabeled_mask]).nonzero(as_tuple=True)[0]
    else:
        training_dataset = dataset
        validation_dataset = dataset
        test_dataset = dataset
        refined_training_idx = training_mask.nonzero(as_tuple=True)[0]
        refined_validation_idx = validation_mask.nonzero(as_tuple=True)[0]

    if return_idx:
        return training_dataset, validation_dataset, test_dataset, refined_training_idx, refined_validation_idx, refined_unlabeled_idx
    return training_dataset, validation_dataset, test_dataset
    
    
    
    
    
def make_ogbn_dataset_splits(dataset_name, splits_root="splits", device="cpu"):
    assert dataset_name in ["ogbn-arxiv", "ogbn-products"]
    dataset = PygNodePropPredDataset(name = dataset_name, root = "datasets/", transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]))

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    suffix = torch.tensor(np.random.randint(0, 100_000_000))
    split_dict = {"training": train_idx, "validation": valid_idx,
                  "test": test_idx, "suffix": suffix}
    split_name = TensorHash.hash_tensor_dict(split_dict)
    try:
        os.makedirs(splits_root, exist_ok=True)
        torch.save(split_dict, os.path.join(splits_root, SplitNaming().name(dataset_name.replace("-", "_"), split_name)))
    except RuntimeError as e:
        print(f"Error saving the split: {e}")

    return {
        "training_idx": train_idx, 
        "validation_idx": valid_idx,
        "test_idx": test_idx,
        "split_name": split_name,
    }

def load_ogbn_dataset_splits(dataset_name, split_code, inductive, dataset_root="datasets/", splits_root="splits", device="cpu"):

    assert dataset_name in  ["ogbn-arxiv", 'ogbn-products']
    arxiv_dataset = PygNodePropPredDataset(name = dataset_name, root = dataset_root, transform=T.Compose([T.ToSparseTensor()])) 
    # # dataset with largest connected component preprocessing
    # dataset = arxiv_dataset[0].to(device)
    # dataset_info = ConfigDict()
    # dataset_info.n_features = arxiv_dataset.num_features
    # dataset_info.n_classes = arxiv_dataset.num_classes
    # dataset_info.n_nodes = arxiv_dataset.data.x.shape[0]
    # dataset_info.dataset_name = dataset_name

    # # largest connected component
    # attr_matrix = dataset.x.cpu().numpy()
    # # convert adj_matrix to scipy sparse matrix
    # edge_index = dataset.adj_t.to_torch_sparse_coo_tensor().coalesce().indices()
    # adj_matrix = to_scipy_sparse_matrix(edge_index, num_nodes=dataset.x.shape[0])
    # labels = dataset.y
    # graphs = SparseGraph(adj_matrix, attr_matrix, labels)

    # prep_graphs = largest_connected_components(sparse_graph=graphs, n_components=1, make_undirected=True)

    # dataset.x = torch.tensor(prep_graphs.attr_matrix)
    # dataset.edge_index = torch.tensor(prep_graphs.get_edgeid_to_idx_array().T, dtype=torch.long)
    # dataset.y = prep_graphs.labels.clone().detach()
    # dataset_info.n_nodes = dataset.x.shape[0]
    # num_nodes = dataset_info.n_nodes

    # dataset without largest connected component preprocessing
    dataset = arxiv_dataset[0]
    num_nodes = dataset.num_nodes
    num_features = dataset.x.shape[1]
    
    dataset_info = ConfigDict()
    dataset_info.n_features = num_features
    dataset_info.n_classes = len(dataset.y.unique())
    dataset_info.n_nodes = num_nodes
    dataset_info.dataset_name = dataset_name

    split_dict = torch.load(os.path.join(splits_root, SplitNaming().name(dataset_name.replace("-", "_"), split_code)))
    training_mask = torch.zeros(num_nodes, dtype=torch.bool)
    training_mask[split_dict["training"]] = True
    validation_mask = torch.zeros(num_nodes, dtype=torch.bool)
    validation_mask[split_dict["validation"]] = True
    unlabeled_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[split_dict["test"]] = True

    assert (training_mask | validation_mask | unlabeled_mask | test_mask).min().item()
    assert (training_mask & validation_mask).max().item() == False
    assert (training_mask & test_mask).max().item() == False
    assert (validation_mask & test_mask).max().item() == False

    dataset.adj_t = dataset.adj_t.to_symmetric()
    if inductive:
        training_dataset = node_induced_subgraph_arxiv(dataset, training_mask | unlabeled_mask)
        validation_dataset = node_induced_subgraph_arxiv(dataset, training_mask | validation_mask | unlabeled_mask)
        test_dataset = dataset
    else:
        training_dataset = dataset
        validation_dataset = dataset
        test_dataset = dataset
        
        training_dataset.edge_index = training_dataset.adj_t.to_torch_sparse_coo_tensor().coalesce().indices()
        validation_dataset.edge_index = training_dataset.adj_t.to_torch_sparse_coo_tensor().coalesce().indices()
        test_dataset.edge_index = training_dataset.adj_t.to_torch_sparse_coo_tensor().coalesce().indices()
            
        training_dataset.y = training_dataset.y.reshape(-1) 
        validation_dataset.y = validation_dataset.y.reshape(-1) 
        test_dataset.y = test_dataset.y.reshape(-1)     
    training_graph = SparseGraph(adj_matrix=to_scipy_sparse_matrix(
        training_dataset.edge_index, num_nodes=num_nodes),
        attr_matrix=training_dataset.x.cpu().numpy(), 
        labels=training_dataset.y)
    training_attr = torch.FloatTensor(training_graph.attr_matrix).to(device)
    training_adj = sparse_tensor(training_graph.adj_matrix.tocoo()).to(device)

    validation_graph = SparseGraph(adj_matrix=to_scipy_sparse_matrix(
        validation_dataset.edge_index, num_nodes=num_nodes),
        attr_matrix=validation_dataset.x.cpu().numpy(), 
        labels=validation_dataset.y)
    validation_attr = torch.FloatTensor(validation_graph.attr_matrix).to(device)
    validation_adj = sparse_tensor(validation_graph.adj_matrix.tocoo()).to(device)

    test_graph = SparseGraph(adj_matrix=to_scipy_sparse_matrix(
        test_dataset.adj_t.to_torch_sparse_coo_tensor().coalesce().indices(), num_nodes=num_nodes),
        attr_matrix=test_dataset.x.cpu().numpy(), 
        labels=test_dataset.y.squeeze())
    test_attr = torch.FloatTensor(test_graph.attr_matrix).to(device)
    test_adj = sparse_tensor(test_graph.adj_matrix.tocoo()).to(device)

    labels = training_dataset.y.to(device)

    return {
        "training_attr": training_attr, 
        "training_adj": training_adj, 
        "validation_attr": validation_attr,
        "validation_adj": validation_adj,
        "test_attr": test_attr, 
        "test_adj": test_adj, 
        "labels": labels, 
        "training_idx": split_dict["training"], 
        "validation_idx": split_dict["validation"], 
        "unlabeled_mask": unlabeled_mask,
        "test_mask": test_mask, 
        "split_name": split_code,
        "dataset_info": dataset_info
    } 
    