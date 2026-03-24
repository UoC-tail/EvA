import os
from torch_geometric.datasets import CitationFull, Planetoid
from ml_collections import ConfigDict
from torch_geometric.utils import to_scipy_sparse_matrix
import json
import torch
from torch.nn import functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import is_undirected

import logging
logger = logging.getLogger(__name__)
logger.propagate = True
logger.setLevel(logging.DEBUG)

from gnn_setup.attacks.prbcd import PRBCD
from gnn_setup.attacks.prbcd_constrained import LRBCD
from gnn_setup.attacks.pgd import PGD
from gnn_setup.attacks.pga import PGA
from gnn_setup.attacks.sga import SGA
from gnn_setup.attacks.dice import DICE
from gnn_setup.attacks.local_dice import LocalDICE
from gnn_setup.attacks.nettack import Nettack
from gnn_setup.attacks.greedy_rbcd import GreedyRBCD
from gnn_setup.setups.data import load_attr_adj
from gnn_setup.utils.robust_training_utils import count_edges_for_idx, count_edge_for_inter_subset
from gnn_setup.utils.metrics import accuracy_from_data as accuracy
from gnn_setup.utils.storage import attack_storage_label
from gnn_setup.conformal.core import ConformalClassifier as CP
from gnn_setup.conformal.scores import APSScore, TPSScore

from eva.core.evattack  import EvAttack
from eva.core.global_attack.subgraph_evattack import EvAttackSubgraph
from eva.core.accelerated import EvAttackAccelerated
from eva.core.accelerated_debug import EvAttackAcceleratedDebug
from eva.core.accelerated_tanh import EvAttackAcceleratedTanh
# from eva.core.accelerated_cross_entropy import EvAttackAcceleratedCE
from eva.core.accelerated_multi_objective import EvAttackAcceleratedMultiObj
from eva.cert.eva_cert import EvaCertAttack
from eva.core.local import EvaLocal
from eva.cert.eva_cp import EvaConformal
import torch_geometric

def load_attack_class(attack_name):
    if attack_name.lower() == "prbcd":
        return PRBCD
    if attack_name.lower() == "lrbcd":
        return LRBCD
    if attack_name.lower() == "pgd":
        return PGD
    if attack_name.lower() == "evattack":
        return EvAttack
    if attack_name.lower() == "evattacksubgraph":
        return EvAttackSubgraph
    if attack_name.lower() == "evattackaccelerated":
        return EvAttackAccelerated
    if attack_name.lower() == "evattackaccelerateddebug":
        return EvAttackAcceleratedDebug
    if attack_name.lower() == "evalocal":
        return EvaLocal
    if attack_name.lower() == "dice":
        return DICE
    if attack_name.lower() == "localdice":
        return LocalDICE
    if attack_name.lower() == "nettack":
        return Nettack
    if attack_name.lower() == "evaconformal":
        return EvaConformal

    if attack_name.lower() == "evattackacceleratedtanh":
        return EvAttackAcceleratedTanh
    if attack_name.lower() == "evattackacceleratedce":
        return EvAttackAcceleratedCE
    if attack_name.lower() == "scaledeva":
        return EvAttackAcceleratedMultiObj
    if attack_name.lower() == "sga":
        return SGA
    if attack_name.lower() == "nettack":
        return Nettack
    if attack_name.lower() == "greedy_rbcd":
        return GreedyRBCD
    if attack_name.lower() == "pga":
        return PGA
    
    
    

    # if attack_name.lower() == "evafast":
    #     return EvaFast
    # if attack_name.lower() == "evalocal":
    #     return EvaLocal
    # if attack_name.lower() == "evatarget":
    #     return EvaTarget
    # if attack_name.lower() == "evasgcpoisoningattack":
        # return EvaSGCPoisoningAttack

def load_cert_attack_class(attack_name):
    if attack_name.lower() == "evacertattack":
        return EvaCertAttack

def create_attack(attack_name, attack_configs, epsilon,
                  dataset, training_idx, unlabeled_idx, test_idx, 
                  model, model_storage_name, split_name, 
                  inductive=False, run_on_subgraph=False,
                  reports_root="./reports", save=True, device="cpu", 
                  debug_active=False, overwrite_n_edges=None, suffix="", control_nodes=None, surrogate_model=None):
    
    attack_idx = test_idx if inductive else torch.concat([test_idx, unlabeled_idx], dim=0).sort()[0] 


    attack_cls = load_attack_class(attack_name)
    
    # TODO: we should check the run on subgraph option:
    # In case of True, it first creates the two hop neighborhood subgraph and then runs the attack on it.

    
    if dataset.get("dataset_info") is not None:
        if dataset["dataset_info"].dataset_name == "ogbn-arxiv":
            test_attr, test_adj, labels = dataset["test_attr"], dataset["test_adj"], dataset["labels"]
        else:
            raise ValueError("The dataset is not ogbn-arxiv")
    else:
        test_attr, test_adj = load_attr_adj(dataset, attack_idx, device=device)
        labels = dataset.y
    # else:
    #     test_attr, test_adj, labels = dataset["test_attr"], dataset["test_adj"], dataset["labels"]

    # test_attr, test_adj = load_attr_adj(dataset, attack_idx, device=device)

    if debug_active:
        attack_configs.update({"debug_active": True})
    if surrogate_model is not None:
        adversary = attack_cls(
        attr=test_attr, adj=test_adj, labels=labels, 
        model=surrogate_model, idx_attack=attack_idx.cpu().numpy(), # why numpy?
        device=device, data_device=device, make_undirected=True, binary_attr=False, 
        training_idx=training_idx, control_nodes=control_nodes, **attack_configs)
    else:
        adversary = attack_cls(
            attr=test_attr, adj=test_adj, labels=labels, 
            model=model, idx_attack=attack_idx.cpu().numpy(), # why numpy?
            device=device, data_device=device, make_undirected=True, binary_attr=False, 
            training_idx=training_idx, control_nodes=control_nodes, **attack_configs)
        
    n_feasible_edges = count_edges_for_idx(test_adj.cpu(), attack_idx.cpu())
    n_attack_edge = (n_feasible_edges * epsilon).int().item() // 2
    print(f"n_attack_edge: {n_attack_edge}")
    if overwrite_n_edges is not None:
        n_attack_edge = overwrite_n_edges

    adversary.attack(n_attack_edge)
    pert_adj, pert_attr = adversary.get_pertubations()

    if isinstance(pert_adj, SparseTensor): # Our final datatype is always torch.tensor with sparse coo layout
        row, col, val = pert_adj.coo()
        if val is None:
            val = torch.ones(row.shape[0], dtype=torch.float, device=device)
        pert_adj = torch.sparse_coo_tensor(indices=torch.stack([row, col]), values=val, size=pert_adj.sizes())

    adv_attacked_edges = (pert_adj - test_adj).coalesce().values().sum().item()


    assert is_undirected((pert_adj - test_adj).coalesce().indices())
    assert adv_attacked_edges <= n_attack_edge * 2

    pert_acc = accuracy(
        model=model, attr=pert_attr, adj=pert_adj, 
        labels=labels.to(pert_attr.device), evaluation_mask=attack_idx.to(pert_attr.device))
    
    attack_artifacts = {
        "pert_adj": pert_adj,
        "pert_attr": pert_attr,
        "pert_acc": pert_acc,
        "adv_attacked_edges": adv_attacked_edges,
        "n_attack_edges": n_attack_edge,
        "attack_configs": attack_configs,
        "attack_name": attack_name,
        "epsilon": epsilon if overwrite_n_edges is None else -1,        
    }
    if debug_active and hasattr(adversary, "debug_info"):
        attack_artifacts.update({"debug_info": adversary.debug_info})

    attack_storage_name = attack_storage_label(
        attack_name=attack_name, model_storage_name=model_storage_name,  
        attack_configs=attack_configs, budget=overwrite_n_edges or epsilon, 
        split_name=split_name, suffix=suffix
    )
    
    try:
        if save:
            os.makedirs(reports_root, exist_ok=True)
            torch.save(attack_artifacts, os.path.join(reports_root, attack_storage_name + ".pt"))
    except RuntimeError as e:
        logger.error(f"Error saving the attack: {e}")
    attack_artifacts.update({"attack_obj": adversary, "attack_storage_name": attack_storage_name})
    return attack_artifacts



def create_attack_scaled(attack_name, attack_configs, epsilon,
                  dataset, training_idx, unlabeled_idx, test_idx, 
                  model, model_storage_name, split_name, 
                  inductive=False, run_on_subgraph=False,
                  reports_root="./reports", save=True, device="cpu", 
                  debug_active=False, overwrite_n_edges=None, suffix="", control_nodes=None, subgraph_size=1):
    
    attack_idx = test_idx if inductive else torch.concat([test_idx, unlabeled_idx], dim=0).sort()[0] 
    
    attack_idx_org = attack_idx.clone()


        
    if dataset.get("dataset_info") is not None:
        if dataset["dataset_info"].dataset_name in ["ogbn-arxiv", 'ogbn-products']:
            test_attr, test_adj, labels = dataset["test_attr"], dataset["test_adj"], dataset["labels"]
        else:
            raise ValueError("The dataset is not ogbn-arxiv")
    else:
        test_attr, test_adj = load_attr_adj(dataset, attack_idx, device=device)
        labels = dataset.y
    
    
    
    test_adj_org = test_adj.clone()
    
    # import pdb; pdb.set_trace()
    # degree = 1
    # n_nodes = test_attr.shape[0]
    # degrees = torch_geometric.utils.degree(test_adj_org.indices()[0,:], num_nodes=n_nodes).to(device)
    # attack_degrees = degrees[attack_idx]
    # attack_idx_cls = attack_idx[attack_degrees !=1]
  
    # test_degrees = torch_geometric.utils.degree(
    # )
    
    steps =( len(attack_idx_org) // subgraph_size)+ 1
    
    print(f"steps: {steps}")
    # sum_all=0
    # n_feasible_edges = count_edges_for_idx(test_adj_org.cpu(), attack_idx_org.cpu())
    # n_attack_edge = ((n_feasible_edges * epsilon).int().item() // 2) // steps
    # import pdb; pdb.set_trace()
    for step in range(steps):
        print(f"-------------------Step {step}-------------------")
        attack_idx = attack_idx_org[step*subgraph_size:(step+1)*subgraph_size] if step != steps - 1 else attack_idx_org[step*subgraph_size:]
            
        attack_cls = load_attack_class(attack_name)
        
        # TODO: we should check the run on subgraph option:
        # In case of True, it first creates the two hop neighborhood subgraph and then runs the attack on it.

        # else:
        #     test_attr, test_adj, labels = dataset["test_attr"], dataset["test_adj"], dataset["labels"]

        # test_attr, test_adj = load_attr_adj(dataset, attack_idx, device=device)

        if debug_active:
            attack_configs.update({"debug_active": True})

        adversary = attack_cls(
            attr=test_attr, adj=test_adj, labels=labels, 
            model=model, idx_attack=attack_idx.cpu().numpy(), # why numpy?
            device=device, data_device=device, make_undirected=True, binary_attr=False, 
            training_idx=training_idx, control_nodes=control_nodes, **attack_configs)
        

        if overwrite_n_edges is not None:
            print(f"-----------Overwriting n_edges: {overwrite_n_edges}-----------")
            n_attack_edge = overwrite_n_edges
        
        n_feasible_edges = count_edges_for_idx(test_adj_org.cpu(), attack_idx.cpu()) - count_edge_for_inter_subset(test_adj_org.cpu(), attack_idx_org.cpu(), attack_idx.cpu())
        n_attack_edge = ((n_feasible_edges * epsilon).int().item() // 2) 
        print(f"step: {step}, n_attack_edge: {n_attack_edge}")

        adversary.attack(n_attack_edge)
        pert_adj, pert_attr = adversary.get_pertubations()

        test_adj = pert_adj.clone()

    if isinstance(pert_adj, SparseTensor): # Our final datatype is always torch.tensor with sparse coo layout
        row, col, val = pert_adj.coo()
        pert_adj = torch.sparse_coo_tensor(indices=torch.stack([row, col]), values=val, size=pert_adj.sizes())

    adv_attacked_edges = (pert_adj - test_adj_org).coalesce().values().sum().item()

    n_feasible_edges = count_edges_for_idx(test_adj_org.cpu(), attack_idx_org.cpu())
    n_attack_edge = (n_feasible_edges * epsilon).int().item() // 2
    assert is_undirected((pert_adj - test_adj_org).coalesce().indices())
    assert adv_attacked_edges <= n_attack_edge * 2

    pert_acc = accuracy(model=model, attr=pert_attr, adj=pert_adj, labels=labels.to(pert_attr.device), evaluation_mask=attack_idx_org.to(pert_attr.device))
    
    attack_artifacts = {
        "pert_adj": pert_adj,
        "pert_attr": pert_attr,
        "pert_acc": pert_acc,
        "adv_attacked_edges": adv_attacked_edges,
        "n_attack_edges": n_attack_edge,
        "attack_configs": attack_configs,
        "attack_name": attack_name,
        "epsilon": epsilon if overwrite_n_edges is None else -1,        
    }
    if debug_active and hasattr(adversary, "debug_info"):
        attack_artifacts.update({"debug_info": adversary.debug_info})

    attack_storage_name = attack_storage_label(
        attack_name=attack_name, model_storage_name=model_storage_name,  
        attack_configs=attack_configs, budget=overwrite_n_edges or epsilon, 
        split_name=split_name, suffix=suffix
    )
    
    try:
        if save:
            os.makedirs(reports_root, exist_ok=True)
            torch.save(attack_artifacts, os.path.join(reports_root, attack_storage_name + ".pt"))
    except RuntimeError as e:
        logger.error(f"Error saving the attack: {e}")
    attack_artifacts.update({"attack_obj": adversary, "attack_storage_name": attack_storage_name})
    return attack_artifacts

def create_certificate_attack(attack_name, attack_configs, epsilon,
                              dataset, training_idx, unlabeled_idx, test_idx, 
                              model, model_storage_name, split_name, 
                              overwrite_n_edges=None,
                              inductive=True, run_on_subgraph=False,
                              reports_root="./reports", save=True, device="cpu", mode="acc",
                              certificate_configs=None, debug_active=False, suffix=""):
    
    attack_idx = test_idx if inductive else torch.concat([test_idx, unlabeled_idx], dim=0).sort()[0]

    attack_cls = load_cert_attack_class(attack_name)

    test_attr, test_adj = load_attr_adj(dataset, attack_idx, device=device)

    if debug_active:
        attack_configs.update({"debug_active": True})

    adversary = attack_cls(
        attr=test_attr, adj=test_adj, labels=dataset.y, 
        model=model, idx_attack=attack_idx.cpu().numpy(), # why numpy?
        device=device, data_device=device, make_undirected=True, binary_attr=False, 
        certificate_configs=certificate_configs, mode=mode,
        training_idx=training_idx, **attack_configs)
    
    n_feasible_edges = count_edges_for_idx(test_adj.cpu(), attack_idx.cpu())
    n_attack_edge = (n_feasible_edges * epsilon).int().item() // 2
    if overwrite_n_edges is not None:
        n_attack_edge = overwrite_n_edges

    adversary.attack(n_attack_edge)
    pert_adj, pert_attr = adversary.get_pertubations()
    statistics = adversary.return_stats()

    pert_acc = statistics.get("perturbed").get("certified_acc")
    

    if isinstance(pert_adj, SparseTensor): # Our final datatype is always torch.tensor with sparse coo layout
        row, col, val = pert_adj.coo()
        pert_adj = torch.sparse_coo_tensor(indices=torch.stack([row, col]), values=val, size=pert_adj.sizes())

    adv_attacked_edges = (pert_adj - test_adj).coalesce().values().sum().item()


    assert is_undirected((pert_adj - test_adj).coalesce().indices())
    assert adv_attacked_edges <= n_attack_edge * 2

    
    attack_artifacts = {
        "pert_adj": pert_adj,
        "pert_attr": pert_attr,
        "pert_acc": pert_acc,
        "adv_attacked_edges": adv_attacked_edges,
        "n_attack_edges": n_attack_edge,
        "attack_configs": attack_configs,
        "attack_name": attack_name,
        "statistics": statistics,
        "epsilon": epsilon if overwrite_n_edges is None else -1,
        "mode": mode,
        "certificate_configs": certificate_configs       
    }
    if debug_active:
        attack_artifacts.update({"debug_info": adversary.debug_info})

    combined_attack_configs = attack_configs.copy()
    combined_attack_configs.update({"certificate": certificate_configs, "mode": mode})

    attack_storage_name = attack_storage_label(
        attack_name=attack_name, model_storage_name=model_storage_name,  
        attack_configs=combined_attack_configs, budget=overwrite_n_edges or epsilon, 
        split_name=split_name, suffix=suffix
    )
    
    try:
        if save:
            os.makedirs(reports_root, exist_ok=True)
            torch.save(attack_artifacts, os.path.join(reports_root, attack_storage_name + ".pt"))
    except RuntimeError as e:
        logger.error(f"Error saving the attack: {e}")
    attack_artifacts.update({"attack_obj": adversary, "attack_storage_name": attack_storage_name, 
                             "certificate_configs": certificate_configs, "mode": mode})
    return attack_artifacts



def create_conformal_attack(attack_name, attack_configs, epsilon,
                  dataset, training_idx, unlabeled_idx, test_idx, 
                  model, model_storage_name, split_name, 
                  inductive=False, run_on_subgraph=False,
                  reports_root="./reports", save=True, device="cpu", mode="coverage", 
                  debug_active=False, overwrite_n_edges=None, suffix=""):
    
    attack_idx = test_idx if inductive else torch.concat([test_idx, unlabeled_idx], dim=0).sort()[0] 


    attack_cls = load_attack_class(attack_name)
    
    # TODO: we should check the run on subgraph option:
    # In case of True, it first creates the two hop neighborhood subgraph and then runs the attack on it.

    
    if dataset.get("dataset_info") is not None:
        if dataset["dataset_info"].dataset_name == "ogbn-arxiv":
            test_attr, test_adj, labels = dataset["test_attr"], dataset["test_adj"], dataset["labels"]
        else:
            raise ValueError("The dataset is not ogbn-arxiv")
    else:
        test_attr, test_adj = load_attr_adj(dataset, attack_idx, device=device)
        labels = dataset.y
    # else:
    #     test_attr, test_adj, labels = dataset["test_attr"], dataset["test_adj"], dataset["labels"]

    # test_attr, test_adj = load_attr_adj(dataset, attack_idx, device=device)

    if debug_active:
        attack_configs.update({"debug_active": True})

    unlabeled_cal_mask = get_cal_mask(unlabeled_idx, fraction=0.3)
    cal_idx = unlabeled_idx[unlabeled_cal_mask].clone()
    cal_mask = torch.zeros((dataset.x.shape[0], ), dtype=bool)
    cal_mask[cal_idx] = True
    eval_mask = torch.zeros((dataset.x.shape[0], ), dtype=bool)
    eval_mask[attack_idx] = True

    conformal = CP([TPSScore()])
    with torch.no_grad():
        model.eval()
        pred = model(test_attr, test_adj)
    scores = conformal.get_scores_from_logits(pred)
    y_true_mask = F.one_hot(labels, num_classes=scores.shape[1]).bool()
    quantile_val = conformal.calibrate_from_scores(scores[cal_mask], y_true_mask[cal_mask])
    pred_set = conformal.predict_from_scores(scores[eval_mask])
    clean_coverage = conformal.coverage(pred_set, y_true_mask[eval_mask])
    clean_set_size = pred_set.sum(1).float().mean().item()

    adversary = attack_cls(
        attr=test_attr, adj=test_adj, labels=labels, 
        model=model, idx_attack=attack_idx.cpu().numpy(), # why numpy?
        device=device, data_device=device, make_undirected=True, binary_attr=False, unlabeled_idx=unlabeled_idx, mode=mode,
        training_idx=training_idx, **attack_configs)
    
    n_feasible_edges = count_edges_for_idx(test_adj.cpu(), attack_idx.cpu())
    n_attack_edge = (n_feasible_edges * epsilon).int().item() // 2
    if overwrite_n_edges is not None:
        n_attack_edge = overwrite_n_edges

    adversary.attack(n_attack_edge)
    pert_adj, pert_attr = adversary.get_pertubations()

    if isinstance(pert_adj, SparseTensor): # Our final datatype is always torch.tensor with sparse coo layout
        row, col, val = pert_adj.coo()
        pert_adj = torch.sparse_coo_tensor(indices=torch.stack([row, col]), values=val, size=pert_adj.sizes())

    adv_attacked_edges = (pert_adj - test_adj).coalesce().values().sum().item()


    assert is_undirected((pert_adj - test_adj).coalesce().indices())
    assert adv_attacked_edges <= n_attack_edge * 2

    with torch.no_grad():
        model.eval()
        pert_pred = model(pert_attr, pert_adj)
    
    pert_scores = conformal.get_scores_from_logits(pert_pred)
    pert_set = pert_scores[eval_mask] > quantile_val
    pert_coverage_old = conformal.coverage(pert_set, y_true_mask[eval_mask])
    pert_set_size_old = pert_set.sum(1).float().mean().item()
    
    pert_quantile_val = conformal.calibrate_from_scores(pert_scores[cal_mask], y_true_mask[cal_mask])
    pert_pred_set = pert_scores[eval_mask] >= pert_quantile_val
    pert_coverage_new = conformal.coverage(pert_pred_set, y_true_mask[eval_mask])
    pert_set_size_new = pert_pred_set.sum(1).float().mean().item()


    pert_acc = accuracy(
        model=model, attr=pert_attr, adj=pert_adj, 
        labels=labels.to(pert_attr.device), evaluation_mask=attack_idx.to(pert_attr.device))
    
    attack_artifacts = {
        "pert_adj": pert_adj,
        "pert_attr": pert_attr,
        "pert_acc": pert_acc,
        "adv_attacked_edges": adv_attacked_edges,
        "n_attack_edges": n_attack_edge,
        "attack_configs": attack_configs,
        "attack_name": attack_name,
        "epsilon": epsilon if overwrite_n_edges is None else -1,
        "clean_coverage": clean_coverage,
        "clean_set_size": clean_set_size,
        "pert_coverage_old": pert_coverage_old,
        "pert_set_size_old": pert_set_size_old,
        "pert_coverage_new": pert_coverage_new,
        "pert_set_size_new": pert_set_size_new,
        "quantile_val": quantile_val,
        "pert_quantile_val": pert_quantile_val,
    }
    if debug_active and hasattr(adversary, "debug_info"):
        attack_artifacts.update({"debug_info": adversary.debug_info})

    attack_storage_name = attack_storage_label(
        attack_name=attack_name, model_storage_name=model_storage_name,  
        attack_configs=attack_configs, budget=overwrite_n_edges or epsilon, 
        split_name=split_name, suffix=suffix
    )
    
    try:
        if save:
            os.makedirs(reports_root, exist_ok=True)
            torch.save(attack_artifacts, os.path.join(reports_root, attack_storage_name + ".pt"))
    except RuntimeError as e:
        logger.error(f"Error saving the attack: {e}")
    attack_artifacts.update({"attack_obj": adversary, "attack_storage_name": attack_storage_name})
    return attack_artifacts


def get_cal_mask(vals_tensor, fraction=0.1):
    perm = torch.randperm(vals_tensor.shape[0])
    mask = torch.zeros((vals_tensor.shape[0]), dtype=bool)
    cutoff_index = int(vals_tensor.shape[0] * fraction)
    mask[perm[:cutoff_index]] = True
    return mask