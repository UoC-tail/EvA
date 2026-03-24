class Naming(object):
    def __init__(self, **kwargs):
        self._suffix = kwargs.get("suffix", "")

    def suffix(self):
        if self._suffix == "":
            return ""
        else:
            return f"-{self._suffix}"

    def name(**kwargs):
        return "-".join([f"{k}_{v}" for k, v in kwargs.items()])
    


class SplitNaming(Naming):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def name(self, dataset_name, split_name):
        return f"{dataset_name}-{split_name}{self.suffix()}.pt"
    
    def conf_name(self, dataset_name, split_name):
        return f"{dataset_name}-{split_name}{self.suffix()}-conf.json"
    


# def dataset_split_name(dataset_name, split_name):
#     return f"{dataset_name}-{split_name}.pt"