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
import random
import numpy as np

experiment = Experiment("Arxiv Training")
experiment.logger = logger


from gnn_setup.setups.data import make_arxiv_dataset_splits, load_arxiv_dataset_splits
from gnn_setup.setups.models import make_arxiv_instance, load_arxiv_instance
from gnn_setup.utils.tensors import set_seed

@experiment.config
def default_config():
    dataset_name = "ogbn-arxiv"
    # model_name in ["GPRGNN", "APPNP"]
    model_name = "GCN"
    
    n_runs = 5
    model_configs = None
    inductive = True
    
    wandb_flag = True
    wandb_project = "arxiv-attack"
    wandb_entity = "Sadegh-Research"
    seed= 5


@experiment.automain 
def run(dataset_name, model_name, n_runs, model_configs, inductive,
        wandb_flag, wandb_project, wandb_entity, seed):

    logger.info("experiment configs:" + str(locals()))

    if wandb_flag:
        wandb.init(project=wandb_project, entity= wandb_entity)
        wandb.config.update(locals())

    
    set_seed(seed)
    # Loading general configs (like dataset_root, etc.) and initial parameters
    general_config = yaml.safe_load(open("./conf/general-config.yaml"))
    default_model_configs = yaml.safe_load(open("./conf/model-configs.yaml")).get("arxiv_configs")
    

    # extracting directory paths
    dataset_root = general_config.get("dataset_root", "data/")
    splits_root = general_config.get("splits_root", "splits/")
    models_root = general_config.get("models_root", "models/")
    splits_root += "/arxiv/"
    models_root += "/arxiv/"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # assert dataset_name == "ogbn-arxiv"
    logger.info("Experiment Started")

    try:
        dataset_splits = [split_record for split_record in os.listdir(splits_root) 
                          if split_record.split("-")[0] == dataset_name.replace("-", "_")]
    except FileNotFoundError:
        dataset_splits = []

    creating_splits = max(n_runs - len(dataset_splits), 0)
    print(f"Found {len(dataset_splits)} splits, creating {creating_splits} more splits")
    for i in tqdm(range(creating_splits)):
        # torch.cuda.empty_cache()
        make_arxiv_dataset_splits(dataset_name, splits_root)

    dataset_splits = [split_record for split_record in os.listdir(splits_root) 
                      if split_record.split("-")[0] == dataset_name.replace("-", "_")][:n_runs]
    print(f"Training {model_name} model on {dataset_name} dataset for {n_runs} splits")
    accs = []

    for split in tqdm(dataset_splits):
        split_code = split.split("-")[1].replace(".pt", "")

        dataset = load_arxiv_dataset_splits(
            dataset_name, split_code, inductive=inductive, 
            dataset_root=dataset_root, splits_root=splits_root, device=device)
        
        training_attr = dataset["training_attr"]
        training_adj = dataset["training_adj"]
        validation_attr = dataset["validation_attr"]
        validation_adj = dataset["validation_adj"]
        test_attr = dataset["test_attr"]
        test_adj = dataset["test_adj"]
        labels = dataset["labels"]
        training_idx = dataset["training_idx"]
        validation_idx = dataset["validation_idx"]
        unlabeled_mask = dataset["unlabeled_mask"]
        test_mask = dataset["test_mask"]
        split_name = dataset["split_name"]
        dataset_info = dataset["dataset_info"]

        try:
            model_instance = load_arxiv_instance(
                model_name=model_name, model_params=model_configs, 
                test_attr=test_attr, test_adj=test_adj, labels=labels, test_mask=test_mask, unlabeled_mask=unlabeled_mask,
                split_name=split_name, dataset_info=dataset_info, inductive=inductive, models_root=models_root,
                default_model_configs=default_model_configs, device=device)
        except FileNotFoundError as e:
            logger.info("Creating model from scratch")
            model_instance = make_arxiv_instance(
                model_name=model_name, model_params=model_configs, dataset_info=dataset_info, 
                training_attr=training_attr, training_adj=training_adj, 
                validation_attr=validation_attr, validation_adj=validation_adj,
                labels=labels, training_idx=training_idx, validation_idx=validation_idx,
                test_attr=test_attr, test_adj=test_adj, test_mask=test_mask, unlabeled_mask=unlabeled_mask,
                inductive=inductive, split_name=split_name,
                models_root=models_root, 
                default_model_configs=default_model_configs, 
                device=device)
        acc = model_instance["accuracy"]
        model_storage_name = model_instance["model_storage_name"]
        print(f"Accuracy: {acc}")
        if wandb_flag:
            wandb.log({"accuracy": acc, "model_storage_name": {model_storage_name}})
        
        accs.append(acc)
    logger.info("Experiment Finished")
    mean_clean_acc = sum(accs)/len(accs)
    std_clean_acc = torch.std(torch.tensor(accs))
    print(f"Average accuracy: {sum(accs)/len(accs)}, with standard deviation: {torch.std(torch.tensor(accs))}")
    if wandb_flag:
        wandb.log({"avg_acc": mean_clean_acc, "std_acc": std_clean_acc})

    logger.info(f"Average accuracy: {sum(accs)/len(accs)}")
    print(f"Average accuracy: {mean_clean_acc}, with standard deviation: {std_clean_acc}")

    # endregion
    