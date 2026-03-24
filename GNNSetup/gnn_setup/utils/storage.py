import os
import json
import torch
from ml_collections import ConfigDict
import hashlib

import logging
logger = logging.getLogger(__name__)
logger.propagate = True
logger.setLevel(logging.DEBUG)

def load_split_files(splits_root, make_if_not_exists=False, **kwargs):
    split_files = []
    
    try:
        for split_file in os.listdir(splits_root):
            if split_file.endswith(".pt"):
                split_files.append(split_file)
    except FileNotFoundError:
        print(f"Directory {splits_root} does not exist.")
        if make_if_not_exists:
            os.makedirs(splits_root)
            print(f"Directory {splits_root} created.")
            return load_split_files(splits_root)

    accepted_args = ["training_nodes", "validation_nodes", "test_nodes", "training_split_type", "validation_split_type", "test_split_type",]

    logging.info(f"current kwargs={kwargs}")

    valid_files = []
    error_files = []

    logging.info(f"There are {len(split_files)} split files in {splits_root}")

    for split_file in split_files:
        logging.info(f"Checking split file {split_file}")
        config_file = split_file.replace(".pt", "-conf.json")
        if not os.path.exists(os.path.join(splits_root, config_file)):
            print(f"Config file {config_file} not found.")
            continue
        try:
            config = json.load(open(os.path.join(splits_root, config_file)))
            logging.info(f"Config file={config}")
            file_accepted = True
            for key_arg in kwargs.keys():
                if key_arg not in accepted_args:
                    pass
                if config.get(key_arg) != kwargs.get(key_arg):
                    file_accepted = False
                    break
            if file_accepted:
                valid_files.append(split_file)
            
        except FileNotFoundError:
            print(f"Error loading config file {config_file}")
            error_files.append(split_file)

    return valid_files

    

class TensorHash(object):
    # Adapted from https://stackoverflow.com/questions/74805446/how-to-hash-a-pytorch-tensor
    MULTIPLIER = 6364136223846793005
    INCREMENT = 1
    MODULUS = 2**64
    def __init__(self):
        pass

    @staticmethod
    def hash_str(x, len=10):
        return str(hex(int(hashlib.sha256(x.encode()).hexdigest(), 16))[-1 * len:])

    @staticmethod
    def hash_tensor(x: torch.Tensor, return_hex=True) -> torch.Tensor:
        assert x.dtype == torch.int64
        while x.ndim > 0:
            x = TensorHash._reduce_last_axis(x)
        if return_hex:
            return hex(x.item())
        return x.item()
    
    @staticmethod
    def hash_tensor_dict(tensor_dict, return_hex=True):
        serialized_dict = (".".join([k + str(TensorHash.hash_tensor(v, return_hex=False)) 
                                     for k, v in tensor_dict.items() if k != "config"]))
        return (TensorHash.hash_str(serialized_dict)) if return_hex else TensorHash.hash_str(serialized_dict)

    @staticmethod
    @torch.no_grad()
    def _reduce_last_axis(x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.int64
        acc = torch.zeros_like(x[..., 0])
        for i in range(x.shape[-1]):
            acc *= TensorHash.MULTIPLIER
            acc += TensorHash.INCREMENT
            acc += x[..., i]
            # acc %= MODULUS  # Not really necessary.
        return acc
    
    @staticmethod
    def hash_model_params(model_name, model_params):
        if isinstance(model_params, ConfigDict):
            return (TensorHash.hash_str(json.dumps([model_name, model_params.to_dict()])))
        return (TensorHash.hash_str(json.dumps([model_name, model_params])))


def model_storage_label(model_name, model_params, dataset_info, inductive, 
                        split_name, self_training=False, robust_training=False, 
                        train_attack_name='PRBCD', robust_epsilon=0.0, suffix=""):
    model_config_hash = TensorHash.hash_model_params(model_name, model_params)
    setting_ind = "ind" if inductive else "tr"
    setting_st = "-st" if self_training else ""
    setting_rt = "-rt_"+train_attack_name+'_'+str(robust_epsilon).replace('.', '_') if robust_training else ""
    return f"{model_name}-{model_config_hash}-{setting_ind}{setting_st}{setting_rt}-{dataset_info.dataset_name}-{split_name}-{suffix}"


def attack_storage_label(attack_name, model_storage_name,  
                         attack_configs, budget, split_name, suffix=""):
    attack_config_hash = TensorHash.hash_model_params(model_name=attack_name, model_params=attack_configs)
    return f"{attack_name}-{attack_config_hash}-eps_{str(budget).replace('.', '_')}-{model_storage_name}-{suffix}"