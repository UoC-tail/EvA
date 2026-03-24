# TODO: Here it is going to be the same evattack function however it just restricts to the 2hop neighborhood.
from eva.core.evattack import EvAttack 
from gnn_setup.data.graph import receptive_field_subgraph, return_subgraph_to_original


class EvAttackSubgraph(EvAttack):
    def __init__(self, k_hops=2, **kwargs):
        super().__init__(**kwargs)
        self.k_hops = k_hops

        self.original_attr = self.attr.clone()
        self.original_adj = self.adj.clone()
        self.original_labels = self.labels.clone()
        self.original_attack_idx = self.idx_attack.copy()
        self.original_training_idx = self.training_idx.clone()
        self.original_mask_attack = self.mask_attack.clone()
        self.n_nodes_original = self.attr.shape[0]

        subgraph_details = receptive_field_subgraph(self.attr, self.adj, self.labels, self.idx_attack, self.k_hops)

        attr_filtered, adj_filtered, labels_filtered, filter_idx_map, filter_mask = subgraph_details
        self.attr = attr_filtered
        self.adj = adj_filtered
        self.labels = labels_filtered
        self.filter_idx_map = filter_idx_map
        self.filter_mask = filter_mask
        self.idx_attack = self.filter_idx_map[self.original_attack_idx]
        self.training_idx = self.filter_idx_map[self.original_training_idx]
        self.training_idx = self.training_idx[self.training_idx >= 0]
        self.mask_attack = self.mask_attack[self.filter_mask.cpu()]
        self.n_nodes = self.attr.shape[0]
    
    def _attack(self, n_perturbations: int, **kwargs):
        super()._attack(n_perturbations, **kwargs)
        attr_adversary_recovered, adj_adversary_recovered, labels_recovered = return_subgraph_to_original(
            attr_reduced=self.attr_adversary, adj_reduced=self.adj_adversary,
            attr_original=self.original_attr, adj_original=self.original_adj,
            filter_idx_map=self.filter_idx_map, filter_mask=self.filter_mask,
            labels_reduced=self.labels, labels_original=self.original_labels,
        )
        self.attr_adversary = attr_adversary_recovered
        self.adj_adversary = adj_adversary_recovered

        print("I am here.")

    