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

from sacred import Experiment

experiment = Experiment("AttackConformal")
experiment.logger = logger

from gnn_setup.utils.configs_manager import refine_dataset_configs, refine_model_configs, refine_attack_configs
from gnn_setup.utils.storage import load_split_files
from gnn_setup.setups.data import make_dataset_split, check_split_valid, load_dataset_split, load_dataset
from gnn_setup.setups.models import load_model_class
from gnn_setup.setups.models import make_trained_model, load_trained_model
from gnn_setup.setups.models import load_robust_model

from gnn_setup.setups.attack import create_attack, create_conformal_attack
from gnn_setup.utils.tensors import set_seed
import wandb

@experiment.config
def default_config():
    # General configs: dataset name, model name, etc.
    dataset_name = "cora_ml"
    model_name = "GPRGNN"
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

    attack_name = "EvaConformal"
    # attack_name = "PRBCD"
    attack_configs = None
    # attack_configs = {"capacity":10, "num_population": 256, "n_steps": 250}
    attack_configs = {"n_steps": 100}
    epsilon = 0.15
    debug_active = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    # for robust model
    robust_model_flag = False
    self_training= False
    robust_training = False
    train_attack_name = "PRBCD"
    robust_epsilon = 0.2
    mode="coverage"


    wandb_flag = False 
    wandb_project = "attack_conformal"
    wandb_entity = "Sadegh-Research"
    seed= 10
    

@experiment.automain 
def run(
    dataset_name, model_name, attack_name, n_runs, inductive,
    training_nodes, validation_nodes, test_nodes, 
    training_split_type, validation_split_type, test_split_type, 
    model_configs, attack_configs, epsilon, debug_active,
    wandb_flag, wandb_project, wandb_entity, seed, robust_model_flag, self_training, 
    robust_training, train_attack_name, robust_epsilon, mode, device):
    
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
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    dataset_splits = dataset_splits[:]
    
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
        # TODO: test if model configs are passed correctly from the experiment config.

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
            
        if model_instance is None:
            print("Model not found. Please train the model first.")
            continue
        
        model = model_instance["model"]
        acc = model_instance["accuracy"]
        model_configs = model_instance["model_configs"]
        model_storage_name = model_instance["model_storage_name"]
        logger.info(f"Model's accuracy: {acc} -- stored under the name: {model_storage_name}")
        clean_accs.append(acc)
        attack_instance = create_conformal_attack(
            attack_name=attack_name, attack_configs=attack_configs, epsilon=epsilon,
            dataset=dataset, training_idx=training_idx, unlabeled_idx=unlabeled_idx, test_idx=test_idx, 
            model=model, model_storage_name=model_storage_name, split_name=split_name, 
            inductive=inductive, debug_active=debug_active,
            reports_root=reports_root, save=True, device=device, mode=mode)
        
        pert_accs.append(attack_instance["pert_acc"])
        logger.info(f"Attack's accuracy: {attack_instance['pert_acc']}")
        logger.info(f"Attack's file = {attack_instance['attack_storage_name']}")
        if wandb_flag:
            wandb.log({
                "mode": mode,
                "clean_acc": acc,
                "pert_acc": attack_instance["pert_acc"],
                "attack_file": attack_instance["attack_storage_name"],
                "clean_coverage": attack_instance["clean_coverage"],
                "pert_coverage": attack_instance["pert_coverage_new"],
                "clean_set_size": attack_instance["clean_set_size"],
                "pert_set_size": attack_instance["pert_set_size_new"],
                "pert_coverage_old": attack_instance["pert_coverage_old"],
                "pert_set_size_old": attack_instance["pert_set_size_old"],
                })


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
