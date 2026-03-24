import os
import yaml
from ml_collections import ConfigDict
from tqdm import tqdm
from copy import deepcopy

import torch
import torch_geometric


import logging
logging.basicConfig(filename='std.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = True
import time
from sacred import Experiment

experiment = Experiment("AttackVanillaModel", save_git_info=False)
experiment.logger = logger

from gnn_setup.utils.configs_manager import refine_dataset_configs, refine_model_configs, refine_attack_configs
from gnn_setup.utils.storage import load_split_files
from gnn_setup.setups.data import make_dataset_split, check_split_valid, load_dataset_split, load_dataset
from gnn_setup.setups.models import load_model_class
from gnn_setup.setups.models import make_trained_model, load_trained_model
from gnn_setup.setups.models import load_robust_model

from gnn_setup.setups.attack import create_attack, create_attack_scaled
from gnn_setup.utils.tensors import set_seed
import wandb



@experiment.config
def default_config():
    # General configs: dataset name, model name, etc.
    dataset_name = "cora_ml"
    model_name = "GCN"
    n_runs = 5 # TODO: previously it was n_splits.
    inductive = True 

    # Configs for splits 
    training_nodes = None # number of training nodes (if integer it should be per-class)
    validation_nodes = None 
    training_split_type = None # it is either "stratified" or "non-stratified"
    validation_split_type = None
    test_nodes = None
    test_split_type = None

    model_configs = None # it is a dictionary of model parameters
    retrain_models = False

    attack_name = "EvAttackAccelerated"
    attack_configs = None
    epsilon = 0.1
    debug_active = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    # for robust model
    robust_model_flag = False
    self_training= True
    robust_training = True
    train_attack_name = "PRBCD"
    robust_epsilon = 0.2
    surrogate_model_name = None 


    wandb_flag = True 
    wandb_project = "attack_models-Neurips"
    wandb_entity = "Sadegh-Research"
    seed= 10
    overwrite_n_edges=None
    
    subgraph_size= 500000
    

@experiment.automain 
def run(
    dataset_name, model_name, attack_name, n_runs, inductive,
    training_nodes, validation_nodes, test_nodes, 
    training_split_type, validation_split_type, test_split_type, 
    model_configs, attack_configs, epsilon, debug_active,
    wandb_flag, wandb_project, wandb_entity, seed, robust_model_flag, self_training, robust_training, train_attack_name, robust_epsilon, device, overwrite_n_edges, subgraph_size, surrogate_model_name):
    
    if wandb_flag:
        wandb.init(project=wandb_project, entity=wandb_entity)
        wandb.config.update(locals())
    
    logger.info("experiment configs:" + str(locals()))

    set_seed(seed)

    # Loading general configs (like dataset_root, etc.) and initial parameters
    general_config = yaml.safe_load(open("./conf/general-config.yaml"))
    default_dataset_configs = yaml.safe_load(open("./conf/data-configs.yaml")).get("configs").get("default")
    default_model_configs = yaml.safe_load(open("./conf/model-configs.yaml")).get("configs")
    default_attack_configs = yaml.safe_load(open("./conf/attack-configs.yaml")).get("configs")

    # extracting directory paths
    dataset_root = general_config.get("dataset_root", "data/")
    splits_root = general_config.get("splits_root", "splits/")
    models_root = general_config.get("models_root", "models/")
    reports_root = general_config.get("reports_root", "reports/")

    logger.info("Experiment Started")

    # region Loading dataset and splits
    refined_dataset_configs = refine_dataset_configs(
        dataset_defaults=default_dataset_configs, 
        training_nodes=training_nodes, validation_nodes=validation_nodes, test_nodes=test_nodes, 
        training_split_type=training_split_type, validation_split_type=validation_split_type, test_split_type=test_split_type)

    training_nodes = refined_dataset_configs["training_nodes"]
    validation_nodes = refined_dataset_configs["validation_nodes"]
    test_nodes = refined_dataset_configs["test_nodes"]
    training_split_type = refined_dataset_configs["training_split_type"]
    validation_split_type = refined_dataset_configs["validation_split_type"]
    test_split_type = refined_dataset_configs["test_split_type"]

    dataset_split_files = load_split_files(
        splits_root=splits_root, make_if_not_exists=True, dataset_name=dataset_name,
        training_nodes=training_nodes, validation_nodes=validation_nodes, test_nodes=test_nodes,
        training_split_type=training_split_type, validation_split_type=validation_split_type, 
        test_split_type=test_split_type,) 
    logger.info("Found {} split files".format(len(dataset_split_files)))

    dataset_splits = [load_dataset_split(
        dataset_name=dataset_name, split_name=split_name, dataset_root=dataset_root, splits_root=splits_root, device=device
    ) for split_name in dataset_split_files]

    if len(dataset_splits) < n_runs:
        raise ValueError("No. Runs = {} is greater than available splits = {}".format(n_runs, len(dataset_splits)))
    dataset_splits = dataset_splits[:n_runs]
    
    dataset, dataset_info = load_dataset(dataset_name, dataset_root)
    # endregion

    model_configs = refine_model_configs(model_name=model_name, model_defaults=default_model_configs, 
                                        model_configs=model_configs,  dataset_info=dataset_info)
    attack_configs = refine_attack_configs(attack_name=attack_name, attack_defaults=default_attack_configs, 
                                           attack_configs=attack_configs)
    if wandb_flag:
        wandb.config.update({"attack_configs": attack_configs}, allow_val_change=True)
    clean_accs = []
    pert_accs = []

    for split in dataset_splits:
        # region loading the split
        
        training_idx = split["training_idx"].to(device)
        validation_idx = split["validation_idx"].to(device)
        test_idx = split["test_idx"].to(device)
        unlabeled_idx = split["unlabeled_idx"].to(device)
        dataset_info = split["dataset_info"]
        split_name = split["split_name"]
        split_config = split["config"]
        # endregion
        # region Preparing the model
        if robust_model_flag:
            model_instance = load_robust_model(model_name=model_name, model_configs=model_configs, 
                                                dataset=dataset, dataset_info=dataset_info,
                                                training_idx=training_idx, validation_idx=validation_idx, unlabeled_idx=unlabeled_idx, test_idx=test_idx,
                                                split_name=split_name, inductive=inductive, model_root=models_root, device=device,
                                                self_training=self_training, robust_training=robust_training, train_attack_name=train_attack_name,
                                                robust_epsilon=robust_epsilon)
        else:
            model_instance = load_trained_model(
                model_name=model_name, model_configs=model_configs,
                dataset=dataset, dataset_info=dataset_info,
                training_idx=training_idx, validation_idx=validation_idx, unlabeled_idx=unlabeled_idx, test_idx=test_idx,
                split_name=split_name, inductive=inductive, model_root=models_root, device=device)
        
        surrogate_model = None
        if surrogate_model_name is not None:
            surrogate_model_config = yaml.safe_load(open(f"./conf/model-configs.yaml")).get("configs").get(surrogate_model_name, {})
            surrogate_model_config = refine_model_configs(model_name=surrogate_model_name, model_defaults=default_model_configs, 
                                        model_configs=surrogate_model_config,  dataset_info=dataset_info)
            surrogate_model = load_trained_model(model_name= surrogate_model_name, model_configs=surrogate_model_config, 
                                                 dataset=dataset, dataset_info=dataset_info,
                                                 training_idx=training_idx, validation_idx=validation_idx, unlabeled_idx=unlabeled_idx, test_idx=test_idx,
                                                 split_name=split_name, inductive=inductive, model_root=models_root, device=device)
            surrogate_model = surrogate_model["model"]
        
        if model_instance is None:
            raise ValueError("Model not found. Please train the model first.")
        model = model_instance["model"]
        model = model.to(device)
        acc = model_instance["accuracy"]
        model_configs = model_instance["model_configs"]
        model_storage_name = model_instance["model_storage_name"]
        logger.info(f"Model's accuracy: {acc} -- stored under the name: {model_storage_name}")
        clean_accs.append(acc)
        start= time.time()
     
        attack_instance = create_attack(
            attack_name=attack_name, attack_configs=attack_configs, epsilon=epsilon,
            dataset=dataset, training_idx=training_idx, unlabeled_idx=unlabeled_idx, test_idx=test_idx, 
            model=model, model_storage_name=model_storage_name, split_name=split_name, 
            inductive=inductive, debug_active=debug_active,
            reports_root=reports_root, save=True, device=device, overwrite_n_edges=overwrite_n_edges, surrogate_model=surrogate_model)
        
        end = time.time()
        
        pert_accs.append(attack_instance["pert_acc"])
        logger.info(f"Attack's accuracy: {attack_instance['pert_acc']}")
        logger.info(f"Attack's file = {attack_instance['attack_storage_name']}")
        pert_acc= attack_instance["pert_acc"]
        print(f"pert_acc{pert_acc}")
        if wandb_flag:
            wandb.log({"clean_acc": acc, "pert_acc": attack_instance["pert_acc"], "attack_file": attack_instance["attack_storage_name"], 'wall_time':end-start, "max_mem":torch.cuda.max_memory_allocated(device='cuda')})


    mean_clean_acc = sum(clean_accs) / len(clean_accs)
    mean_pert_acc = sum(pert_accs) / len(pert_accs)
    std_clean_acc = torch.tensor(clean_accs).std()
    std_pert_acc = torch.tensor(pert_accs).std()
    if wandb_flag:
        wandb.log({"avg_clean_acc": mean_clean_acc, "avg_pert_acc": mean_pert_acc, "std_clean_acc": std_clean_acc, "std_pert_acc": std_pert_acc})
    logger.info(f"Average clean accuracy: {mean_clean_acc}, with standard deviation: {std_clean_acc}")
    print(f"Average clean accuracy: {mean_clean_acc}, with standard deviation: {std_clean_acc}")
    logger.info(f"Average perturbed accuracy: {mean_pert_acc}, with standard deviation: {std_pert_acc}")
    print(f"Average perturbed accuracy: {mean_pert_acc}, with standard deviation: {std_pert_acc}")
    logger.info("Experiment Finished")
    try:
        wandb.finish()
    except Exception as e:
        print("wandb.finish failed:", e)
    finally:
        # As a fallback: force exit if threads hang
        import os, sys
        os._exit(0)

