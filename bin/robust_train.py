import os
import yaml
from tqdm import tqdm
from copy import deepcopy

import torch
from gnn_setup.utils.configs_manager import refine_dataset_configs, refine_model_configs
from gnn_setup.utils.storage import load_split_files
from gnn_setup.setups.data import load_dataset_split, load_dataset
from gnn_setup.setups.models import make_robust_model, load_robust_model
from gnn_setup.utils.tensors import set_seed
from sacred import Experiment
import logging
import wandb
logging.basicConfig(filename='std.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = True

experiment = Experiment("RobustGNNTraining")
experiment.logger = logger


@experiment.config
def default_config():
    ## Experiment configs
    dataset_name = "citeseer"

    # model_name in ["GCN", "DenseGCN", "GAT", "GPRGNN", "DenseGPRGNN", "APPNP", "ChebNetII", "SoftMedian_GDC"]
    model_name = "GCN"
    n_runs = 5

    # Configs for splits
    training_nodes = None # number of training nodes (if integer it should be per-class)
    validation_nodes = None 
    training_split_type = None # it is either "stratified" or "non-stratified"
    validation_split_type = None
    test_nodes = None
    test_split_type = None

    model_configs = None

    # attack_name in ["PRBCD", "LRBCD", "EvaAttack", "EVAFAST", "PGD"] 
    train_attack_name = "PRBCD"
    loss_type = 'tanhMargin'
    
    inductive = True
    self_training = True
    robust_training = True
    robust_epsilon = 0.2
    
    validate_every = None

    train_configs = None
    train_attack_configs = None
    val_attack_configs = None

    wandb_flag = True
    wandb_project = "EAV-robust-train"
    wandb_entity = "Sadegh-Research"
    seed= 10

    retrain_models = False
    save_models = True


@experiment.automain
def run(dataset_name, model_name, n_runs, training_nodes, 
        validation_nodes, training_split_type, validation_split_type, 
        test_nodes, test_split_type, train_configs, validate_every,
        model_configs, train_attack_name, train_attack_configs, val_attack_configs , 
        inductive, self_training ,robust_training, robust_epsilon,
        retrain_models, save_models,
        wandb_flag, wandb_project, wandb_entity, seed):
    

    set_seed(seed)

    if wandb_flag:
        wandb.init(project=wandb_project, entity= wandb_entity)
        wandb.config.update(locals())
    
    logger.info("experiment configs:" + str(locals()))

    # region Loading the configurations and defaults
    ## Loading general configs (like dataset_root, etc.) and initial parameters
    general_config = yaml.safe_load(open("./conf/general-config.yaml"))
    default_dataset_configs = yaml.safe_load(open("./conf/data-configs.yaml")).get("configs").get("default")
    default_model_configs = yaml.safe_load(open("./conf/model-configs.yaml")).get("configs")
    default_robust_model_configs = yaml.safe_load(open("./conf/robust-configs.yaml")).get("configs")

    # extracting directory paths
    dataset_root = general_config.get("dataset_root", "data/")
    splits_root = general_config.get("splits_root", "splits/")
    models_root = general_config.get("models_root", "models/")

    additional_train_configs = deepcopy(train_configs or dict())
    train_configs = deepcopy(default_robust_model_configs.get("training", dict(
            lr=1e-2,
            weight_decay=1e-3,
            patience=200,
            max_epochs=3000
        )))
    train_configs.update(additional_train_configs)
    
    if validate_every is None:
        if train_attack_name in ["LRBCD", "PRBCD"]:
            validate_every = 1
        elif train_attack_name in ["EvAttackAccelerated"]:
            validate_every = 10

    
    val_attack_name = train_attack_name
    default_train_attack_configs = default_robust_model_configs.get("train_attack_configs").get(train_attack_name, None)
    default_val_attack_configs = default_robust_model_configs.get("val_attack_configs").get(val_attack_name, None)
    if default_train_attack_configs is None:
        raise(ValueError(f"{train_attack_name} Invalid attack name"))
    if default_val_attack_configs is None:
        raise(ValueError(f"{val_attack_name} Invalid attack name"))
    additional_train_attack_configs = deepcopy(train_attack_configs or dict())
    additional_val_attack_configs = deepcopy(val_attack_configs or dict())
    train_attack_configs = deepcopy(default_train_attack_configs)
    val_attack_configs = deepcopy(default_val_attack_configs)
    train_attack_configs.update(additional_train_attack_configs)
    val_attack_configs.update(additional_val_attack_configs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # endregion

    logger.info("Experiment Started")

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
        if not retrain_models:
            model_instance = load_robust_model(model_name=model_name, model_configs=model_configs, 
                                                dataset=dataset, dataset_info=dataset_info,
                                                training_idx=training_idx, validation_idx=validation_idx, unlabeled_idx=unlabeled_idx, test_idx=test_idx,
                                                split_name=split_name, inductive=inductive, model_root=models_root, device=device,
                                                self_training=self_training, robust_training=robust_training, train_attack_name=train_attack_name,
                                                robust_epsilon=robust_epsilon)
            logger.info("Model not found, creating the robust model from scratch")
        else:
            model_instance = None

        if model_instance is None:
            print("Model not found. Run training scripts to train the model.")
            model_instance = make_robust_model(model_name=model_name, model_configs=model_configs,
                                                dataset=dataset, dataset_info=dataset_info, inductive=inductive,
                                                training_idx=training_idx, validation_idx=validation_idx, unlabeled_idx=unlabeled_idx, test_idx=test_idx,
                                                split_name=split_name, model_root=models_root, device=device,
                                                self_training=self_training, robust_training=robust_training, train_attack_name=train_attack_name, val_attack_name=val_attack_name,
                                                robust_epsilon=robust_epsilon, validate_every= validate_every, train_configs=train_configs,
                                                train_attack_configs=train_attack_configs, val_attack_configs=val_attack_configs,
                                                save=save_models)
            

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
        clean_acc = model_instance["accuracy"]
        clean_accs.append(clean_acc)


    mean_clean_acc = torch.mean(torch.tensor(clean_accs))
    std_clean_acc = torch.std(torch.tensor(clean_accs))
    if wandb_flag:
        wandb.log({"avg_acc": mean_clean_acc, "std_acc": std_clean_acc, "model_storage_name": model_storage_name})
    
    print(f"Mean clean accuracy on clean dateset: {mean_clean_acc:.4f} $\\pm$ {std_clean_acc:.4f}")
    print("Experiment Finished")

