import copy

def refine_dataset_configs(dataset_defaults, **kwargs):
    outvalues = {}
    for key, value in kwargs.items():
        if value is not None:
            outvalues[key] = value
        else:
            outvalues[key] = dataset_defaults.get(key, None)
    return outvalues

def refine_model_configs(model_name, model_defaults, model_configs, dataset_info):
    if model_configs is None:
        model_configs = dict()

    # final_config = copy.deepcopy(model_configs)
    final_config = dict()
    try:
        model_default_configs = model_defaults.get(model_name)
        final_config.update(model_default_configs)
    except:
        raise ValueError(f"Model name {model_name} not found in the default model configs.")
    
    final_config.update(model_configs)
    final_config["n_features"] = dataset_info.get("n_features")
    final_config["n_classes"] = dataset_info.get("n_classes")
    return final_config

def refine_attack_configs(attack_name, attack_defaults, attack_configs):
    if attack_configs is None:
        attack_configs = dict()
    
    final_config = dict()
    try:
        attack_default_configs = attack_defaults.get(attack_name)
        final_config.update(attack_default_configs)
    except:
        raise ValueError(f"Attack name {attack_name} not found in the default attack configs.")
    final_config.update(attack_configs)
    return final_config
