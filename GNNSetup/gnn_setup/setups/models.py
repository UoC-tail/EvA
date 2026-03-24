import logging
logger = logging.getLogger(__name__)
logger.propagate = True
logger.setLevel(logging.DEBUG)

import os
from ml_collections import ConfigDict
import torch
import numpy as np
from tqdm import tqdm
from torch_sparse import SparseTensor

from gnn_setup.gnns.gcn import GCN, DenseGCN
from gnn_setup.gnns.gnnguard import GNNGuard
from gnn_setup.gnns.rgcn import RGCN
from gnn_setup.gnns.sgc import SGC
from gnn_setup.gnns.gprgnn import GPRGNN
from gnn_setup.gnns.gprgnn_dense import DenseGPRGNN
from gnn_setup.gnns.chebynet2 import ChebNetII
from gnn_setup.gnns.gat_weighted import GAT
from gnn_setup.gnns.rgnn import RGNN
from gnn_setup.gnns.gcn_svd import GCNSVD
from gnn_setup.gnns.gcn_jaccard import GCNJaccard

from gnn_setup.gnns import create_model, GPRGNN, DenseGPRGNN, ChebNetII
from gnn_setup.gnns.gprgnn import GPR_prop

from gnn_setup.setups.data import load_attr_adj, splited_datasets
from gnn_setup.gnns.helpers.train import train, train_inductive

from gnn_setup.utils.storage import TensorHash, model_storage_label
from gnn_setup.utils.metrics import accuracy_from_data as accuracy

from gnn_setup.utils.robust_training_utils import count_edges_for_idx
from gnn_setup.setups.attack import load_attack_class
import torch.nn.functional as F
from copy import deepcopy
from torch_geometric.utils import is_undirected

def load_model_class(model_name):
    if model_name not in ["GCN", "DenseGCN", "GAT", "GPRGNN", "DenseGPRGNN", "APPNP", "ChebNetII", "SoftMedian_GDC", "GNNGuard", "RGCN", "SGC", "GCNSVD", "GCNJaccard"]:
        raise ValueError("The model config is not in the cofigs file!")
    
    if model_name == 'GCN':
        return GCN
    if model_name == 'DenseGCN':
        return DenseGCN
    if model_name == 'GNNGuard':
        return GNNGuard
    if model_name == 'GAT':
        return GAT
    if model_name == 'GPRGNN':
        return GPRGNN
    if model_name == 'DenseGPRGNN':
        return DenseGPRGNN
    if model_name == 'APPNP':
        return GPRGNN
    if model_name == 'ChebNetII':
        return ChebNetII
    if model_name == 'SoftMedian_GDC':
        return RGNN
    if model_name == 'RGCN':
        return RGCN
    if model_name =="SGC":
        return SGC
    if model_name == "GCNSVD":
        return GCNSVD
    if model_name == "GCNJaccard":
        return GCNJaccard
    
def make_trained_model(model_name, model_configs, 
                       dataset, dataset_info,
                       training_idx, validation_idx, unlabeled_idx, test_idx,
                       split_name, inductive=False, model_root="./models", device="cpu", save=True):
    model_class = load_model_class(model_name)
    model = model_class(**model_configs)
    model.to(device)

    training_dataset, validation_dataset, test_dataset, updated_train_idx, updated_validation_idx, updated_unlabeled_idx = splited_datasets(
        dataset, dataset_info=dataset_info, 
        training_idx=training_idx, validation_idx=validation_idx, test_idx=test_idx, unlabeled_idx=unlabeled_idx,
        inductive=inductive, return_idx=True)
    training_attr, training_adj = load_attr_adj(training_dataset, training_idx, device=device)
    validation_attr, validation_adj = load_attr_adj(validation_dataset, validation_idx, device=device)
    test_attr, test_adj = load_attr_adj(dataset, test_idx, device=device)

    if not inductive:
        training_trace = train(
            model=model, attr=training_attr.to(device), adj=training_adj.to(device), labels=training_dataset.y.to(device),
            idx_train=training_idx, idx_val=validation_idx, display_step=100,
            lr=model_configs.get("lr", None), 
            weight_decay=model_configs.get("weight_decay", None), 
            patience=model_configs.get("patience", None),
            max_epochs=model_configs.get("max_epochs", None),
        )
    else:
        training_trace = train_inductive(
            model=model, attr_training=training_attr.to(device), attr_validation=validation_attr.to(device), 
            adj_training=training_adj.to(device), adj_validation=validation_adj.to(device),
            labels_training=training_dataset.y.to(device), labels_validation=validation_dataset.y.to(device),
            idx_train=updated_train_idx, idx_val=updated_validation_idx, display_step=100,
            lr=model_configs.get("lr", None),
            weight_decay=model_configs.get("weight_decay", None),
            patience=model_configs.get("patience", None),
            max_epochs=model_configs.get("max_epochs", None),
        )

    eval_mask = torch.zeros(dataset_info.n_nodes, dtype=torch.bool)
    eval_mask[test_idx] = True
    if not inductive:
        eval_mask[unlabeled_idx] = True
    acc = accuracy(model, test_attr, test_adj, test_dataset.y.to(device), eval_mask)
    
    model_storage_name = model_storage_label(
        model_name=model_name, 
        model_params=model_configs, 
        dataset_info=dataset_info, 
        inductive=inductive, 
        split_name=split_name)
    try:
        if save:
            os.makedirs(model_root, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_root, f"{model_storage_name}.pt"))
    except RuntimeError as e:
        logger.error(f"Error saving the model: {e}")

    return {
        "model": model,
        "model_configs": model_configs, 
        "accuracy": acc,
        "model_storage_name": model_storage_name,
    }


def load_trained_model(
        model_name, model_configs, 
        dataset, dataset_info,
        training_idx, validation_idx, unlabeled_idx, test_idx,
        split_name, inductive=False, model_root="./models", device="cpu"):
    model_class = load_model_class(model_name)
    model = model_class(**model_configs)
    model.to(device)


    model_storage_name = model_storage_label(
        model_name=model_name, 
        model_params=model_configs, 
        dataset_info=dataset_info, 
        inductive=inductive, 
        split_name=split_name)
    
    try:
        print(f"Loading model from {os.path.join(model_root, f'{model_storage_name}.pt')}")
        state_dict = torch.load(os.path.join(model_root, f"{model_storage_name}.pt"), map_location='cpu')
        model.load_state_dict(state_dict)
    except FileNotFoundError as e:
        logger.error(f"Error loading the model: {e}")
        return None
    eval_idx = test_idx if inductive else torch.cat([test_idx, unlabeled_idx]).sort().values
    eval_mask = torch.zeros(dataset_info.n_nodes, dtype=torch.bool)
    eval_mask[eval_idx] = True
    
    _, _, test_dataset = splited_datasets(
        dataset, dataset_info=dataset_info, 
        training_idx=training_idx, validation_idx=validation_idx, test_idx=test_idx, unlabeled_idx=unlabeled_idx,
        inductive=inductive)

    test_attr, test_adj = load_attr_adj(dataset, eval_idx, device=device)
    model = model.to(device)
    acc = accuracy(model, test_attr.to(device), test_adj.to(device), test_dataset.y.to(device), eval_mask.to(device))
    return {
        "model": model,
        "model_configs": model_configs,
        "accuracy": acc,
        "model_storage_name": model_storage_name,
    }


def make_robust_model(model_name, model_configs, 
                       dataset, dataset_info,
                       training_idx, validation_idx, unlabeled_idx, test_idx,
                       split_name, inductive=False, model_root="./models", device="cpu",
                      self_training=True, robust_training=True, train_attack_name='PRBCD', val_attack_name='PRBCD',
                      train_attack_configs=None, val_attack_configs=None,
                      robust_epsilon=0.2, validate_every= 10, train_configs=None, save=False):
    
    model_class = load_model_class(model_name)
    model = model_class(**model_configs)
    model.to(device)

    training_dataset, validation_dataset, test_dataset, updated_train_idx, updated_validation_idx, updated_unlabeled_idx = splited_datasets(
        dataset, dataset_info=dataset_info, 
        training_idx=training_idx, validation_idx=validation_idx, test_idx=test_idx, unlabeled_idx=unlabeled_idx,
        inductive=inductive, return_idx=True)
    # updated_validation_idx = updated_validation_idx.to(device)
    training_attr, training_adj = load_attr_adj(training_dataset, training_idx, device=device)
    validation_attr, validation_adj = load_attr_adj(validation_dataset, validation_idx, device=device)
    test_attr, test_adj = load_attr_adj(dataset, test_idx, device=device)
    # region pretrain
    if not inductive:
        training_trace = train(
            model=model, attr=training_attr.to(device), adj=training_adj.to(device), labels=training_dataset.y.to(device),
            idx_train=training_idx, idx_val=validation_idx, display_step=100,
            lr=model_configs.get("lr", None), 
            weight_decay=model_configs.get("weight_decay", None), 
            patience=model_configs.get("patience", None),
            max_epochs=train_configs.get("pre_train_epochs", None),
        )
    else:
        training_trace = train_inductive(
            model=model, attr_training=training_attr.to(device), attr_validation=validation_attr.to(device), 
            adj_training=training_adj.to(device), adj_validation=validation_adj.to(device),
            labels_training=training_dataset.y.to(device), labels_validation=validation_dataset.y.to(device),
            idx_train=updated_train_idx, idx_val=updated_validation_idx, display_step=100,
            lr=model_configs.get("lr", None),
            weight_decay=model_configs.get("weight_decay", None),
            patience=model_configs.get("patience", None),
            max_epochs=train_configs.get("pre_train_epochs", None),
        )

    if train_attack_name == "PGD" and model_name == "GCN":
        model = from_sparse_GCN(model, model_configs)
    elif train_attack_name == "PGD" and model_name == "GPRGNN":
        model = from_sparse_GPRGNN(model, model_configs)

    if isinstance(model, (GPRGNN, DenseGPRGNN)) and isinstance(model.prop1, GPR_prop) and model.prop1.norm == True: # exclude prop coeffs from weight decay
        print('Excluding GPR-GNN coefficients from weight decay as we use normalization')
        grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if 'prop1.temp' != n],
                    "weight_decay": train_configs['weight_decay'],
                    'lr':train_configs['lr']
                },
                {
                    "params": [p for n, p in model.named_parameters() if 'prop1.temp' == n],
                    "weight_decay": 0.0,
                    'lr':train_configs['lr']
                },
            ]
        optimizer = torch.optim.Adam(grouped_parameters)
    elif isinstance(model, ChebNetII):
        optimizer = torch.optim.Adam([
            {'params': model.lin1.parameters(), 'weight_decay': train_configs['weight_decay'], 'lr': train_configs['lr']},
            {'params': model.lin2.parameters(), 'weight_decay': train_configs['weight_decay'], 'lr': train_configs['lr']},
            {'params': model.prop1.parameters(), 'weight_decay': model.prop_wd, 'lr': model.prop_lr}])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=train_configs['lr'], weight_decay=train_configs['weight_decay'])
    # endregion

    # region self training
    if self_training:
        baseline_model = load_model_class(model_name)(**model_configs).to(device)
        if not inductive:
            training_trace = train(
                model=baseline_model, attr=training_attr, adj=training_adj, labels=training_dataset.y.to(device), idx_train=training_idx, idx_val=validation_idx, 
                display_step=100,
                lr=model_configs.get("lr", None), 
                weight_decay=model_configs.get("weight_decay", None), 
                patience=model_configs.get("patience", None),
                max_epochs=model_configs.get("max_epochs", None)
                )
        else:
            training_trace = train_inductive(
                model=baseline_model, attr_training=training_attr, attr_validation=validation_attr, 
                adj_training=training_adj, adj_validation=validation_adj,  labels_training=training_dataset.y.to(device), labels_validation=validation_dataset.y.to(device), 
                idx_train=updated_train_idx, idx_val=updated_validation_idx, display_step=100,
                lr=model_configs.get("lr", None),
                weight_decay=model_configs.get("weight_decay", None),
                patience=model_configs.get("patience", None),
                max_epochs=model_configs.get("max_epochs", None)
                )
            
        logits = baseline_model(training_attr, training_adj)
        pseudolabels = torch.argmax(logits, dim=1)
        pseudolabels[updated_train_idx] = training_dataset.y.to(device)[updated_train_idx]
        train_labels = pseudolabels
    else:
        train_labels = training_dataset.y.to(device)
    # endregion self training

    eval_mask = torch.zeros(dataset_info.n_nodes, dtype=torch.bool)
    eval_mask[test_idx] = True
    if not inductive:
        eval_mask[unlabeled_idx] = True
    clean_acc = accuracy(model, test_attr, test_adj, test_dataset.y.to(device), eval_mask)

    # region robust training
    # init attack adjs 
    adj_attacked_val = validation_adj.detach()
    adj_attacked_train = training_adj.detach()
    # init trace variables
    loss_trace = []
    gamma_trace = []
    best_loss=np.inf

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_configs['lr'], weight_decay=train_configs['weight_decay'])

    adv_train_idx = torch.concat([updated_train_idx, updated_unlabeled_idx], dim=0).sort()[0] if inductive else training_idx
    if robust_training:
        n_train_edges = count_edges_for_idx(training_adj, adv_train_idx) # num edges connected to train nodes
        n_perturbations_train = (robust_epsilon * n_train_edges).int().item() // 2

        n_val_edges = count_edges_for_idx(validation_adj, updated_validation_idx) # num edges connected to val nodes
        n_perturbations_val = (robust_epsilon * n_val_edges).int().item() // 2

    # training loop
    for it in tqdm(range(train_configs['max_epochs']), desc='Training...'):
        adv_train_idx = torch.concat([updated_train_idx, updated_unlabeled_idx], dim=0).sort()[0] if inductive else training_idx 

        # Generate adversarial adjacency
        if robust_epsilon > 0:
            model.eval()
            attack_cls = load_attack_class(train_attack_name)
            adversary = attack_cls(
                attr=training_attr, adj=training_adj, labels=training_dataset.y.to(device), 
                model=model, idx_attack=adv_train_idx.cpu().numpy(), # why numpy?
                device=device, data_device=device, make_undirected=True, binary_attr=False, 
                training_idx=updated_train_idx, **train_attack_configs)
            
            adversary.attack(n_perturbations_train)
            pert_adj, pert_attr = adversary.get_pertubations()
            if isinstance(pert_adj, SparseTensor): # Our final datatype is always torch.tensor with sparse coo layout
                row, col, val = pert_adj.coo()
                pert_adj = torch.sparse_coo_tensor(indices=torch.stack([row, col]), values=val, size=pert_adj.sizes())

            # adv_attacked_edges = (pert_adj - test_adj).coalesce().values().sum().item() #TODO now this not working for inductive
            # assert is_undirected((pert_adj - test_adj).coalesce().indices())
            # assert adv_attacked_edges <= n_perturbations_train * 2  

            adj_attacked_train = pert_adj
            training_attr = pert_attr
            del adversary

        # train step
        model.train()
        optimizer.zero_grad()
        logits = model(training_attr, adj_attacked_train)
        loss = F.cross_entropy(logits[adv_train_idx], train_labels[adv_train_idx])
        loss.backward()
        optimizer.step()        

        if isinstance(model, GPRGNN) and isinstance(model.prop1, GPR_prop):
            gamma_trace.append(model.prop1.temp.detach().cpu())
        loss_trace.append(loss.item())

        # validation step 
        adv_train_idx = torch.concat([updated_train_idx, updated_validation_idx, updated_unlabeled_idx], dim=0).sort()[0] if inductive else training_idx
        if it % validate_every == 0:
            if robust_epsilon > 0:
                model.eval()
                attack_cls = load_attack_class(val_attack_name)
                adversary = attack_cls(
                    attr=validation_attr, adj=validation_adj, labels=validation_dataset.y.to(device), 
                    model=model, idx_attack=adv_train_idx.cpu().numpy(), # why numpy?
                    device=device, data_device=device, make_undirected=True, binary_attr=False, 
                    training_idx=training_idx, **val_attack_configs)
                
                adversary.attack(n_perturbations_val)
                pert_adj, pert_attr = adversary.get_pertubations()
                if isinstance(pert_adj, SparseTensor): # Our final datatype is always torch.tensor with sparse coo layout
                    row, col, val = pert_adj.coo()
                    pert_adj = torch.sparse_coo_tensor(indices=torch.stack([row, col]), values=val, size=pert_adj.sizes())
                    
                # adv_attacked_edges = (pert_adj - test_adj).coalesce().values().sum().item()
                # assert is_undirected((pert_adj - test_adj).coalesce().indices())
                # assert adv_attacked_edges <= n_perturbations_val * 2

                adj_attacked_val = pert_adj
                validation_attr = pert_attr
                del adversary

            with torch.no_grad():
                model.eval()
                logits_val = model(validation_attr, adj_attacked_val)
                labels = validation_dataset.y.to(device)
                # import pdb; pdb.set_trace()
                loss_val = F.cross_entropy(logits_val[updated_validation_idx], labels[updated_validation_idx])
                # loss_trace_val.append(loss_val.item())

        # save new best model and break if patience is reached
        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = it
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
        else:
            if it >= best_epoch + train_configs['patience']:
                break
    
    # endregion robust training

    model.load_state_dict(best_state)
    model.eval()
    robust_acc = accuracy(model, test_attr, test_adj, test_dataset.y.to(device), eval_mask)
    # region save the model
    model_storage_name = model_storage_label(model_name=model_name,
                                            model_params=model_configs,
                                            dataset_info=dataset_info,
                                            inductive=inductive,
                                            self_training=self_training,
                                            robust_training=robust_training,
                                            train_attack_name=train_attack_name,
                                            robust_epsilon=robust_epsilon,
                                            split_name=split_name)

    try:
        if save:
            os.makedirs(model_root, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_root, f"{model_storage_name}.pt"))
    except RuntimeError as e:
        logger(f"Error saving the model: {e}")
    # endregion
        

    return {
        "model": model,
        "model_configs": model_configs,
        "accuracy": robust_acc,
        "model_storage_name": model_storage_name
    }


def load_robust_model(model_name, model_configs, 
                       dataset, dataset_info,
                       training_idx, validation_idx, unlabeled_idx, test_idx,
                       split_name, inductive=False, model_root="./models", device="cpu",
                      self_training=True, robust_training=True, train_attack_name='PRBCD',
                      robust_epsilon=0.2, save=False):
    
    model_class = load_model_class(model_name)
    model = model_class(**model_configs)
    model.to(device)
    
    # TODO: add a check to see if the model config hash is same as the hash of the current model config
    model_storage_name = model_storage_label(model_name=model_name,
                                            model_params=model_configs,
                                            dataset_info=dataset_info,
                                            inductive=inductive,
                                            self_training=self_training,
                                            robust_training=robust_training,
                                            train_attack_name=train_attack_name,
                                            robust_epsilon=robust_epsilon,
                                            split_name=split_name)
    
    try:
        state_dict = torch.load(os.path.join(model_root, f"{model_storage_name}.pt"), map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError as e:
        logger.error(f"Error loading the model: {e}")
        return None
    
    eval_idx = test_idx if inductive else torch.cat([test_idx, unlabeled_idx]).sort().values
    eval_mask = torch.zeros(dataset_info.n_nodes, dtype=torch.bool)
    eval_mask[eval_idx] = True

    
    _, _, test_dataset = splited_datasets(
        dataset, dataset_info=dataset_info, 
        training_idx=training_idx, validation_idx=validation_idx, test_idx=test_idx, unlabeled_idx=unlabeled_idx,
        inductive=inductive)

    test_attr, test_adj = load_attr_adj(dataset, eval_idx, device=device)
    acc = accuracy(model, test_attr.to(device), test_adj.to(device), test_dataset.y.to(device), eval_mask)

    return {
        "model": model,
        "model_configs": model_configs,
        "model_storage_name": model_storage_name,
        "accuracy": acc
    }

def from_sparse_GCN(sparse_model, args):
    dense_GCN = DenseGCN(**args)
    dense_GCN.activation = deepcopy(sparse_model.activation)
    dense_GCN.layers[0].gcn_0._linear = deepcopy(sparse_model.layers[0].gcn_0.lin)
    dense_GCN.layers[0].activation_0 = deepcopy(sparse_model.layers[0].activation_0)
    dense_GCN.layers[0].dropout_0 = deepcopy(sparse_model.layers[0].dropout_0)
    dense_GCN.layers[1].gcn_1._linear = deepcopy(sparse_model.layers[1].gcn_1.lin)
    return dense_GCN

def from_sparse_GPRGNN(sparse_model, args):
    dense_GPRGNN = DenseGPRGNN(**args)
    dense_GPRGNN.prop1.temp = deepcopy(sparse_model.prop1.temp)
    dense_GPRGNN.lin1 = deepcopy(sparse_model.lin1)
    dense_GPRGNN.lin2 = deepcopy(sparse_model.lin2)
    return dense_GPRGNN



def make_arxiv_instance(model_name, model_params, dataset_info, 
                          training_attr, training_adj, validation_attr, validation_adj,
                          labels, training_idx, validation_idx, 
                          test_attr, test_adj, test_mask, unlabeled_mask, inductive,
                          split_name, models_root="models", default_model_configs=None, suffix='',
                          device='cpu'):
    # Generating the model, and training it on the dataset
    if model_params is None:
        model_params = ConfigDict(default_model_configs.get(model_name))
        model_params.n_features = dataset_info.n_features
        model_params.n_classes = dataset_info.n_classes

    model = load_model_class(model_name)(**model_params.to_dict()).to(device)
    if not inductive:
        training_trace = train(
            model=model, attr=training_attr, adj=training_adj, labels=labels, idx_train=training_idx, idx_val=validation_idx, display_step=100,
            lr=model_params.lr, weight_decay=model_params.weight_decay, patience=model_params.patience,
            max_epochs=model_params.max_epochs)
    else:
        training_trace = train_inductive(
            model=model, attr_training=training_attr, attr_validation=validation_attr, 
            adj_training=training_adj, adj_validation=validation_adj,  labels_training=labels, labels_validation=labels, 
            idx_train=training_idx, idx_val=validation_idx, display_step=100,
            lr=model_params.lr, weight_decay=model_params.weight_decay, patience=model_params.patience,
            max_epochs=model_params.max_epochs)

    eval_mask = test_mask if inductive else (test_mask | unlabeled_mask)
    acc = accuracy(model, test_attr, test_adj, labels, eval_mask)

    model_storage_name = model_storage_label(model_name=model_name, 
                                            model_params=model_params, 
                                            dataset_info=dataset_info, 
                                            inductive=inductive, 
                                            split_name=split_name)

    try: 
        os.makedirs(models_root, exist_ok=True)
        torch.save(model, os.path.join(models_root, f"{model_storage_name}.pt"))
    except RuntimeError as e:
        print(f"Error saving the model: {e}")

    return {
        "model": model,
        "model_params": model_params,
        "accuracy": acc,
        "model_storage_name": model_storage_name
    }


def load_arxiv_instance(model_name, model_params, dataset_info,
                        test_attr, test_adj, labels, test_mask, unlabeled_mask,
                        split_name, inductive=False,
                        models_root="models", self_training=False, robust_training=False, train_attack_name='PRBCD', robust_epsilon=0.2,
                        default_model_configs=None, suffix='', device='cpu'):
    if default_model_configs is None:
        raise ValueError("default_model_configs is required to load the model")
    
    if model_params is None:
        model_params = ConfigDict(default_model_configs.get(model_name))
        model_params.n_features = dataset_info.n_features
        model_params.n_classes = dataset_info.n_classes

    model = load_model_class(model_name)(**model_params.to_dict()).to(device)
    model_storage_name = model_storage_label(model_name=model_name,
                                            model_params=model_params,
                                            dataset_info=dataset_info,
                                            inductive=inductive,
                                            self_training=self_training,
                                            robust_training=robust_training,
                                            train_attack_name=train_attack_name,
                                            robust_epsilon=robust_epsilon,
                                            split_name=split_name)
    
    try:
        model = torch.load(os.path.join(models_root, f"{model_storage_name}.pt")).to(device)
    except RuntimeError as e:
        print(f"Error loading the model: {e}")
        return None
    
    eval_mask = test_mask if inductive else (test_mask | unlabeled_mask)
    acc = accuracy(model, test_attr, test_adj, labels, eval_mask)
    return {
        "model": model,
        "model_params": model_params,
        "model_storage_name": model_storage_name,
        "accuracy": acc
    }

