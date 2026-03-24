import os
import yaml
from tqdm import tqdm
import torch
from copy import deepcopy

import logging
logging.basicConfig(filename='std.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = True

from sacred import Experiment

experiment = Experiment("Arxiv Training", save_git_info=False)
experiment.logger = logger
from gnn_setup.utils.configs_manager import refine_dataset_configs, refine_model_configs, refine_attack_configs
from gnn_setup.setups.data import make_arxiv_dataset_splits, load_arxiv_dataset_splits
from gnn_setup.setups.models import load_arxiv_instance
from gnn_setup.setups.attack import create_attack, create_attack_scaled
from gnn_setup.utils.tensors import set_seed
import wandb


@experiment.config
def default_config():
    dataset_name = "ogbn-arxiv"
    # model_name in ["GPRGNN", "APPNP"]
    model_name = "GCN"
    n_runs = 5

    model_configs = None
    inductive = True

    attack_name = "EvAttackAccelerated"
    # attack_name = "PRBCD"

    attack_configs = None
    epsilon = 0.001

    wandb_flag = True
    wandb_project = "ICLR-arxiv-attack"
    wandb_entity = "Sadegh-Research"
    seed= 5
    subgraph_size= 500
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    

@experiment.automain 
def run(dataset_name, model_name, attack_name, n_runs, inductive,
    model_configs, attack_configs, epsilon,
    wandb_flag, wandb_project, wandb_entity, seed, device, subgraph_size):



    logger.info("experiment configs:" + str(locals()))
    if wandb_flag:
        wandb.init(project=wandb_project, entity= wandb_entity)
        wandb.config.update(locals())

    # Loading general configs (like dataset_root, etc.) and initial parameters
    general_config = yaml.safe_load(open("./conf/general-config.yaml"))
    default_model_configs = yaml.safe_load(open("./conf/model-configs.yaml")).get("arxiv_configs")
    default_attack_configs = yaml.safe_load(open("./conf/attack-configs.yaml")).get("arxiv_configs")

    # extracting directory paths
    dataset_root = general_config.get("dataset_root", "data/")
    splits_root = general_config.get("splits_root", "splits/")
    models_root = general_config.get("models_root", "models/")
    reports_root = general_config.get("reports_root", "reports/")
    splits_root += "/arxiv/"
    models_root += "/arxiv/"
    reports_root += "/arxiv/"


    assert dataset_name == "ogbn-arxiv"
    logger.info("Experiment Started")
    
    # load the dataset splits
    dataset_splits = [split_record for split_record in os.listdir(splits_root) 
                      if split_record.split("-")[0] == dataset_name.replace("-", "_")]
    if len(dataset_splits) < n_runs:
        raise ValueError("No. Runs = {} is greater than available splits = {}".format(n_runs, len(dataset_splits)))
    dataset_splits = dataset_splits[:n_runs]

    print(f"Training {model_name} model on {dataset_name} dataset for {n_runs} splits")
    clean_accs = []
    pert_accs = []

    attack_configs = refine_attack_configs(attack_name=attack_name, attack_defaults=default_attack_configs, 
                                        attack_configs=attack_configs)
    
 
    attack_configs = deepcopy(default_attack_configs[attack_name])
    if attack_configs is not None:
        attack_configs.update(attack_configs)
        
    if wandb_flag:
        wandb.config.update({"attack_configs": attack_configs}, allow_val_change=True)

    for split in tqdm(dataset_splits):
        split_code = split.split("-")[1].replace(".pt", "")

        dataset = load_arxiv_dataset_splits(
            dataset_name, split_code, inductive=inductive, 
            dataset_root=dataset_root, splits_root=splits_root, device=device)
        
        test_attr = dataset["test_attr"]
        test_adj = dataset["test_adj"]
        labels = dataset["labels"]
        training_idx = dataset["training_idx"]
        unlabeled_mask = dataset["unlabeled_mask"]
        unlabeled_idx = unlabeled_mask.nonzero(as_tuple=True)[0]
        test_mask = dataset["test_mask"]
        test_idx = test_mask.nonzero(as_tuple=True)[0]
        split_name = dataset["split_name"]
        dataset_info = dataset["dataset_info"]
    
        try:
            model_instance = load_arxiv_instance(
                model_name=model_name, model_params=model_configs, 
                test_attr=test_attr, test_adj=test_adj, labels=labels, test_mask=test_mask, unlabeled_mask=unlabeled_mask,
                split_name=split_name, dataset_info=dataset_info, inductive=inductive, models_root=models_root,
                default_model_configs=default_model_configs, device=device)
        except FileNotFoundError as e:
            print(e)
            raise ValueError("Model not found. Run training scripts to train the model.")
        
        model = model_instance["model"]
        model = model.to(device)
        model_storage_name = model_instance["model_storage_name"]
        clean_acc = model_instance["accuracy"]
        clean_accs.append(clean_acc)


        attack_instance = create_attack_scaled(attack_name, attack_configs, epsilon, dataset, training_idx, unlabeled_idx, test_idx,  model, model_storage_name, split_name,  inductive=False, device=device, subgraph_size=subgraph_size) #control_nodes=torch.randint(0, 1000, (100,)).unique()
        # attack_instance = create_attack(attack_name, attack_configs, epsilon, dataset, training_idx, unlabeled_idx, test_idx,  model, model_storage_name, split_name,  inductive=False, device=device) #control_nodes=torch.randint(0, 1000, (100,)).unique()

        pert_acc = attack_instance["pert_acc"]
        # import pdb; pdb.set_trace()
        pert_accs.append(pert_acc)
        if wandb_flag:
            wandb.log({"clean accuracy": clean_acc, "pert_accuracy": pert_acc,  "model_storage_name": {model_storage_name}})


    mean_clean_acc = sum(clean_accs) / len(clean_accs)
    mean_pert_acc = sum(pert_accs) / len(pert_accs)
    std_clean_acc = torch.tensor(clean_accs).std()
    std_pert_acc = torch.tensor(pert_accs).std()
    logger.info(f"Average clean accuracy: {mean_clean_acc}, with standard deviation: {std_clean_acc}")
    print(f"Average clean accuracy: {mean_clean_acc}, with standard deviation: {std_clean_acc}")
    logger.info(f"Average perturbed accuracy: {mean_pert_acc}, with standard deviation: {std_pert_acc}")
    print(f"Average perturbed accuracy: {mean_pert_acc}, with standard deviation: {std_pert_acc}")
    if wandb_flag:
        wandb.log({"avg_acc": mean_clean_acc, "std_acc": std_clean_acc, "avg_pert_acc": mean_pert_acc, "std_pert_acc": std_pert_acc}) 
    logger.info("Experiment Finished")