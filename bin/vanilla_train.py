import os
import yaml
from ml_collections import ConfigDict
from tqdm import tqdm

import torch
import torch_geometric
import wandb


import logging
logging.basicConfig(filename='std.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = True

from sacred import Experiment

experiment = Experiment("VanillaGNNTraining")
experiment.logger = logger

from gnn_setup.utils.configs_manager import refine_dataset_configs, refine_model_configs
from gnn_setup.utils.storage import load_split_files
from gnn_setup.setups.data import make_dataset_split, check_split_valid, load_dataset_split, load_dataset
from gnn_setup.setups.models import load_model_class
from gnn_setup.setups.models import make_trained_model, load_trained_model
from gnn_setup.utils.tensors import set_seed

@experiment.config
def default_config():
    # General configs: dataset name, model name, etc.
    dataset_name = "cora_ml"
    model_name = "APPNP"
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
    save_models = True
    wandb_flag = True 
    wandb_project = "EAV-vanila-train"
    wandb_entity = "Sadegh-Research"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed= 10

@experiment.automain 
def run(
    dataset_name, model_name, n_runs, inductive,
    training_nodes, validation_nodes, test_nodes, 
    training_split_type, validation_split_type, test_split_type, 
    model_configs, retrain_models, save_models,
    wandb_flag, wandb_project, wandb_entity, seed, device):
    print("Running experiment with the following configs:")

    set_seed(seed)
    
    logger.info("experiment configs:" + str(locals()))
    if wandb_flag:
        wandb.init(project=wandb_project, entity= wandb_entity)
        wandb.config.update(locals())
    # Loading general configs (like dataset_root, etc.) and initial parameters
    general_config = yaml.safe_load(open("./conf/general-config.yaml"))
    default_dataset_configs = yaml.safe_load(open("./conf/data-configs.yaml")).get("configs").get("default")
    default_model_configs = yaml.safe_load(open("./conf/model-configs.yaml")).get("configs")

    # extracting directory paths
    dataset_root = general_config.get("dataset_root", "data/")
    splits_root = general_config.get("splits_root", "splits/")
    models_root = general_config.get("models_root", "models/")

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
    dataset_splits = dataset_splits[:n_runs]
    
    dataset, dataset_info = load_dataset(dataset_name, dataset_root)
    # endregion
    
    model_configs = refine_model_configs(model_name=model_name, model_defaults=default_model_configs, 
                                        model_configs=model_configs,  dataset_info=dataset_info)

    accs = []
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
        # TODO: test if model configs are passed correctly from the experiment config.

        if not retrain_models:
            model_instance = load_trained_model(
                model_name=model_name, model_configs=model_configs,
                dataset=dataset, dataset_info=dataset_info,
                training_idx=training_idx, validation_idx=validation_idx, unlabeled_idx=unlabeled_idx, test_idx=test_idx,
                split_name=split_name, inductive=inductive, model_root=models_root, device=device)
             
        else:
            model_instance = None
            
        if model_instance is None:
            logger.info("Model not found, creating the model from scratch")
            model_instance = make_trained_model(
                model_name=model_name, model_configs=model_configs,
                dataset=dataset, dataset_info=dataset_info, inductive=inductive,
                training_idx=training_idx, validation_idx=validation_idx, unlabeled_idx=unlabeled_idx, test_idx=test_idx,
                split_name=split_name, model_root=models_root, device=device, save=save_models)

        else: 
            logger.info("Model found, loading the model")

        print("model_accuracy", model_instance["accuracy"])
        if wandb_flag:
            wandb.log({"accuracy": model_instance["accuracy"]})
        model = model_instance["model"]
        acc = model_instance["accuracy"]
        model_configs = model_instance["model_configs"]
        model_storage_name = model_instance["model_storage_name"]
        logger.info(f"Model's accuracy: {acc} -- stored under the name: {model_storage_name}")
        accs.append(acc)

        # endregion
    logger.info("Experiment Finished")
    print(f"Average accuracy: {sum(accs)/len(accs)}, with standard deviation: {torch.std(torch.tensor(accs))}")
    if wandb_flag:
        wandb.log({"avg_acc": sum(accs)/len(accs), "std_acc": torch.std(torch.tensor(accs))})

    logger.info(f"Average accuracy: {sum(accs)/len(accs)}")
    # endregion
    