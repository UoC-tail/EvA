import os
import yaml
from ml_collections import ConfigDict
from tqdm import tqdm

import torch
import torch_geometric

import logging
logging.basicConfig(filename='std.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = True
from sacred import Experiment
experiment = Experiment("MakeGraphSplits")
experiment.logger = logger

from gnn_setup.utils.storage import load_split_files
from gnn_setup.utils.configs_manager import refine_dataset_configs
from gnn_setup.setups.data import make_dataset_split, check_split_valid, load_dataset_split
from gnn_setup.utils.tensors import set_seed

@experiment.config
def default_config():
    # General configs: dataset name, model name, etc.
    dataset_name = "citeseer"
    n_runs = 10 # TODO: previously it was n_splits.

    # Configs for splits
    training_nodes = None # number of training nodes (if integer it should be per-class)
    validation_nodes = None 
    training_split_type = None # it is either "stratified" or "non-stratified"
    validation_split_type = None
    test_nodes = None
    test_split_type = None
    recreate_splits = False

    expr_suffix = "" # TODO: handle the suffix for the other experiments
    seed = 10



@experiment.automain
def run(
    dataset_name, n_runs, 
    training_nodes, validation_nodes, test_nodes, 
    training_split_type, validation_split_type, test_split_type,
    recreate_splits, expr_suffix, seed):
    logger.info("experiment configs:" + str(locals()))

    set_seed(seed)
    ## Loading general configs (like dataset_root, etc.) and initial parameters
    general_config = yaml.safe_load(open("./conf/general-config.yaml"))
    default_dataset_configs = yaml.safe_load(open("./conf/data-configs.yaml")).get("configs").get("default")
    
    # extracting directory paths
    dataset_root = general_config.get("dataset_root", "data/")
    splits_root = general_config.get("splits_root", "splits/")
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info("Experiment Started")
    # Trains the specified model on the given graph and saves the model artifacts, and the splits.

    # region Refining configs given the defaults
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
    # endregion
    
    # region Loading currently available dataset splits
    dataset_split_files = load_split_files(
        splits_root=splits_root, make_if_not_exists=True, dataset_name=dataset_name,
        training_nodes=training_nodes, validation_nodes=validation_nodes, test_nodes=test_nodes,
        training_split_type=training_split_type, validation_split_type=validation_split_type, 
        test_split_type=test_split_type,) 
    logger.info("Found {} split files".format(len(dataset_split_files)))
    
    if recreate_splits:
        dataset_split_files = []  
    
    dataset_splits = [load_dataset_split(
        dataset_name=dataset_name, split_name=split_name, dataset_root=dataset_root, splits_root=splits_root, device=device
    ) for split_name in dataset_split_files]

    logger.info(f"among the {len(dataset_split_files)} splits, {len(dataset_splits)} splits are loaded successfully.")
    # TODO: 2. There should be a validator function to do assert checks on the dataset split.
    
    remaining_splits = max(n_runs - len(dataset_splits), 0)
    logger.info("Found {} splits, creating {} more splits".format(len(dataset_split_files), remaining_splits))
    # endregion

    for i in range(remaining_splits):
        # region Creating dataset splits 
        split = make_dataset_split(
            dataset_name=dataset_name,
            training_nodes=training_nodes,
            validation_nodes=validation_nodes,
            test_nodes=test_nodes,
            test_split_type=test_split_type,
            training_split_type=training_split_type,
            validation_split_type=validation_split_type,
            inductive=True, # in the transductive setting we combine the unlabeled and test set; TODO: check
            dataset_root=dataset_root,
            splits_root=splits_root,
            device=device,
        )
        training_idx = split["training_idx"]
        validation_idx = split["validation_idx"]
        test_idx = split["test_idx"]
        unlabeled_idx = split["unlabeled_idx"]
        dataset_info = split["dataset_info"]
        split_name = split["split_name"]
        split_config = split["config"]

        try:
            check_split_valid(training_idx=training_idx, validation_idx=validation_idx, test_idx=test_idx, 
                unlabeled_idx=unlabeled_idx, dataset_info=dataset_info, split_name=split_name, split_config=split_config)
        except AssertionError as e:
            logger.error("Error in split validation: {}".format(e))
    # endregion
    

