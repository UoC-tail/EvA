from copy import deepcopy

import torch
from evotorch.core import Problem, SolutionBatch
from evotorch.operators.base import CopyingOperator

from eva.utils import triangular_column_indices, triangular_matrix_index
from eva.utils import linear_to_triu_idx, triu_to_linear_idx
from torch_geometric.utils import degree

from tqdm import tqdm
import numpy as np

def temperature_sampling_from_vector(x, idx, num_samples, temperature=1.0):
    """
    Samples elements from tensor x with a temperature-controlled distribution.
    
    Args:
        x (torch.Tensor): 1D tensor containing integer elements.
        num_samples (int): Number of samples to draw.
        temperature (float): Temperature parameter to control uniformity.
                             - temperature = 1.0: Original uniform sampling.
                             - temperature < 1.0: More biased towards higher weights.
                             - temperature > 1.0: More uniform.
                             
    Returns:
        torch.Tensor: Sampled elements.
    """
    if temperature < 1e-6:
        # Effectively uniform sampling
        unique_elements, _ = torch.unique(x, return_counts=True)
        K = unique_elements.size(0)
        p = torch.ones(K) / K
        sampled_unique = torch.multinomial(p, num_samples, replacement=True)
        sampled_elements = unique_elements[sampled_unique]
        idxs= idx[sampled_unique]
    else:
        # Identify unique elements and their counts
        unique_elements, counts = torch.unique(x, return_counts=True)
        
        base_weights = 1.0 / counts.float()
        adjusted_weights = base_weights.pow(1.0 / temperature)
        adjusted_weights = adjusted_weights / adjusted_weights.sum()
        indices = torch.searchsorted(unique_elements, x)
        weights = adjusted_weights[indices]
        
        # Normalize weights for the entire tensor
        weights = weights / weights.sum()
        
        # Sample indices based on the adjusted weights
        sampled_indices = torch.multinomial(weights, num_samples, replacement=False)
        sampled_elements = x[sampled_indices]
        idxs= idx[sampled_indices]
    
    return sampled_elements, idxs

class GraphMutation(CopyingOperator):
    def __init__(self, problem, **kwargs):
        super().__init__(problem)
    
    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        raise NotImplementedError
    
    def update_locals(self, population, **kwargs):
        pass


def safe_degree(row_or_col, num_nodes):
    valid_mask = (row_or_col >= 0)
    return degree(row_or_col[valid_mask], num_nodes=num_nodes)

class LocalBudgetMutation(GraphMutation):
    def __init__(self, problem, delta=0.5, adj=None, n_nodes=None, device=None, idx_attack=None, **kwargs):

        super().__init__(problem)

        self.device = device
        self.delta = delta
        self.adj = adj.coalesce()
        self.n_nodes = n_nodes or self.adj.shape[0]
        edge_index = self.adj.indices()[:, self.adj.values() != 0]
        self.orig_degrees = degree(edge_index[0, :], num_nodes=self.n_nodes)
        self.local_budget = self.delta * self.orig_degrees
        self.last_epoch = False
        if idx_attack is not None:
            self.idx_attack = idx_attack.to(self.device)
        else:
            self.idx_attack = torch.arange(self.n_nodes).to(self.device)
        
    def update_locals(self, population, **kwargs):
        self.last_epoch = kwargs.get("last_epoch")

    @torch.no_grad()
    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        result = deepcopy(batch)
        perturbations = result.access_values()
        if not self.last_epoch:
            corrected_perturbations = self._local_adjacency_filter(perturbations, 'better-naive')
            result.set_values(corrected_perturbations)
        else:
            corrected_perturbations = self._local_adjacency_filter(perturbations)
            result.set_values(corrected_perturbations)
        return result

    @staticmethod
    def mark_violations(perturbations=None, perturbation_rows=None, perturbation_cols=None, n_nodes=None, local_budget=None):
        """
        Returns the violations and the capacities of the nodes. 
        The output is a tuple of two tensors each of shape (n_pop, n_nodes)
        """
        if perturbation_rows is None and perturbation_cols is None:
            perturbation_rows, perturbation_cols = linear_to_triu_idx(n_nodes, perturbations)

        if perturbation_rows.ndim == 2:
            pert_degrees = torch.stack(
                [safe_degree(perturbation_rows[i], num_nodes=n_nodes) for i in range(perturbation_rows.shape[0])]
                ) + torch.stack(
                    [safe_degree(perturbation_cols[i], num_nodes=n_nodes) for i in range(perturbation_rows.shape[0])]
                )
        elif perturbation_rows.ndim == 1:
            pert_degrees = safe_degree(perturbation_rows, num_nodes=n_nodes) + safe_degree(perturbation_cols, num_nodes=n_nodes)
            
        violations = torch.relu(pert_degrees - local_budget).ceil()
        capacities = (torch.relu(local_budget - pert_degrees)).floor()
        return violations, capacities

    def _local_adjacency_filter(self, perturbation, method="projection"):
        device = perturbation.device
        valid_perturbations = perturbation.clone()
        valid_perturbations[valid_perturbations < 0] = 0
        perturbation_rows, perturbation_cols = linear_to_triu_idx(self.n_nodes, valid_perturbations)

        if method == "naive":
            violations, capacities = self.mark_violations(
                perturbation_rows=perturbation_rows, perturbation_cols=perturbation_cols,
                n_nodes=self.n_nodes, local_budget=self.local_budget)

            if violations.sum() <= 0:
                return perturbation
            pert_violation_scores = torch.stack([
                violations[i][perturbation_rows[i]] + violations[i][perturbation_cols[i]]
                for i in range(perturbation.shape[0])], dim=0)

            new_perturbation = perturbation.clone()
            new_perturbation[pert_violation_scores > 0] = -1
            return new_perturbation

        if method == "better-naive":
            new_perturbation = perturbation.clone()
            for i in range(30):
                perturbation_rows, perturbation_cols = linear_to_triu_idx(self.n_nodes, new_perturbation)
                violations, capacities = self.mark_violations(
                    perturbations=new_perturbation,
                    n_nodes=self.n_nodes, local_budget=self.local_budget)

                if violations.sum() <= 0:
                    return new_perturbation
                pert_violation_scores = torch.stack([
                    violations[i][perturbation_rows[i]] + violations[i][perturbation_cols[i]]
                    for i in range(perturbation.shape[0])], dim=0)

                chances = torch.rand_like(pert_violation_scores) < ((pert_violation_scores / pert_violation_scores.max(1).values.reshape(-1, 1))**2)
                new_perturbation[chances] = -1
            
            violations, capacities = self.mark_violations(
                    perturbations=new_perturbation,
                    n_nodes=self.n_nodes, local_budget=self.local_budget)

            new_perturbation[pert_violation_scores > 0] = -1
            return new_perturbation

        if method == "projection":
            new_perturbation = perturbation.clone()
            # idx_values, counts = torch.unique(new_perturbation.flatten(), return_counts=True)
            # counts = (counts / new_perturbation.shape[0]).clamp(0, 1)

            idx_values, frequency = torch.unique(new_perturbation.flatten(), return_counts=True)
            frequency = (frequency / new_perturbation.shape[0] + torch.rand_like(frequency.float()) * 0.05).clamp(0, 1)
            sorted_idxs, mapper = new_perturbation.flatten().sort()
            map_back = mapper.argsort()
            unique_to_sorted = torch.searchsorted(idx_values, sorted_idxs)
            frequency_sorted = frequency[unique_to_sorted]
            pert_scores = frequency_sorted[map_back].reshape(new_perturbation.shape)

            # Until here each perturbation is assigned with a score.
            pert_score_sorted, sorted_pert_idx = pert_scores.sort(dim=1, descending=True)
            perturbations_sorted = torch.gather(new_perturbation, 1, sorted_pert_idx)
            pert_sorted_projected = torch.zeros_like(new_perturbation).bool()
            pert_rows, pert_cols = linear_to_triu_idx(self.n_nodes, perturbations_sorted)

            remaining_budget = torch.stack([self.local_budget.clone() for _ in range(new_perturbation.shape[0])])
            
            for idx in range(perturbation.shape[1]):
                
                edge_budget = torch.minimum(remaining_budget[torch.arange(new_perturbation.shape[0]), pert_rows[:, idx]], 
                                remaining_budget[torch.arange(new_perturbation.shape[0]), pert_cols[:, idx]]) - 1
                violating = (edge_budget < 0)
                pert_sorted_projected[:, idx] = ~violating

                budget_subtract = torch.zeros_like(remaining_budget)
                budget_subtract[torch.arange(new_perturbation.shape[0]), pert_rows[:, idx]] += (~violating).float()
                budget_subtract[torch.arange(new_perturbation.shape[0]), pert_cols[:, idx]] += (~violating).float()
                remaining_budget = remaining_budget - budget_subtract

            perturbations_sorted[~pert_sorted_projected] = -1
            reverse_pert_idx = sorted_pert_idx.argsort()
            new_perturbation = torch.gather(perturbations_sorted, 1, reverse_pert_idx)
            return new_perturbation


class PositiveIntMutation(GraphMutation):
    def __init__(self, problem, mutation_rate=0.1, toggle_rate=0.0):
        super().__init__(problem)
        self.mutation_rate = mutation_rate
        self.toggle_rate = toggle_rate
    
    @torch.no_grad()
    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        result = deepcopy(batch)
        data = result.access_values()
        mutation_mask = torch.rand(size=data.shape, device=data.device) < self.mutation_rate
        mutant_data = data[mutation_mask]
        toggle_mutations = torch.rand(size=mutant_data.shape, device=mutant_data.device) < self.toggle_rate
        new_vals = torch.randint(0, self.problem.upper_bounds, size=mutant_data.shape, device=data.device)
        new_vals[(mutant_data >= 0) & toggle_mutations] = -1
        new_vals[(mutant_data < 0) & (~toggle_mutations)] = -1
        data[mutation_mask] = new_vals
        return result

    def update_locals(self, population, **kwargs):
        pass


class VarinacePreservingMutation(PositiveIntMutation):
    def __init__(self, problem, mutation_rate=0.1, variance_coef=0.1, radius=0.5):
        super().__init__(problem, mutation_rate=mutation_rate)

        self.initial_mutation_rate = mutation_rate
        self.last_variance = -1
        self.population_var = -1
        self.radius = radius
        self.variance_coef = variance_coef

    def update_locals(self, population):
        evals = population.access_evals()
        self.last_variance = self.population_var
        self.population_var = evals.var()

        if self.last_variance == -1:
            return
        if self.population_var <= 1e-6:
            self.mutation_rate = self.initial_mutation_rate + self.initial_mutation_rate * self.radius
            return

        self.mutation_rate = self.mutation_rate + (self.population_var - self.last_variance) / self.last_variance * self.variance_coef

        self.mutation_rate = torch.clamp(self.mutation_rate, 
                                         self.initial_mutation_rate - self.initial_mutation_rate * self.radius, 
                                         self.initial_mutation_rate + self.initial_mutation_rate * self.radius)
        if self.mutation_rate < 0:
            self.mutation_rate = 0

    def statistics(self, population):
        return {"mutation_rate": self.mutation_rate, "variance": self.population_var}
    


class AdaptiveIntMutation(CopyingOperator):
    def __init__(self, problem, mutation_rate=0.1, toggle_rate=0.5, patience=10):
        super().__init__(problem)
        self.mutation_rate = mutation_rate
        self.initial_mutation_rate = mutation_rate
        self.patience = patience
        self.toggle_rate = toggle_rate

        self.min_mutation_rate = mutation_rate / 3
        self.max_mutation_rate = mutation_rate * 3

        self.variances = []
        self.second_previous_variance = -1
        self.variance_derivative = 0
        self.eta = -1 * mutation_rate / 10
    
    def set_variance(self, var):
        self.variances.append(var)

        if len(self.variances) < 10:
            return
        
        var_coef = var - torch.tensor(self.variances[-10]).mean()
        
        self.mutation_rate = self.mutation_rate + (var_coef / self.variances[-10].mean()) * self.mutation_rate
        self.mutation_rate = (self.mutation_rate).item()
        
        self.mutation_rate = min(self.mutation_rate, self.max_mutation_rate)
        self.mutation_rate = max(self.mutation_rate, self.min_mutation_rate)

        if not ((self.mutation_rate <= 0) | (self.mutation_rate >= 0)):
            # number is nan
            self.mutation_rate = self.initial_mutation_rate
    
    @torch.no_grad()
    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        result = deepcopy(batch)
        data = result.access_values()
        mutation_mask = torch.rand(size=data.shape, device=data.device) < self.mutation_rate
        mutant_data = data[mutation_mask]
        toggle_mutations = torch.rand(size=mutant_data.shape, device=mutant_data.device) < self.toggle_rate
        new_vals = torch.randint(0, self.problem.upper_bounds, size=mutant_data.shape, device=data.device)
        new_vals[(mutant_data >= 0) & toggle_mutations] = -1
        new_vals[(mutant_data < 0) & (~toggle_mutations)] = -1
        data[mutation_mask] = new_vals
        return result
    
    




class IdxControl(GraphMutation):
    def __init__(self, problem, idx_attack, n_nodes, mutation_rate=0.1, adversary=None):
        super().__init__(problem)
        self.mutation_rate = mutation_rate
        self.idx_attack =idx_attack.clone() if isinstance(idx_attack, torch.Tensor) else torch.tensor(idx_attack)
        self.n_nodes = n_nodes
        self.adversary = adversary
        self.control_nodes = self.adversary.control_nodes
        self.initial_mutation_rate = mutation_rate
        self.last_variance= -1
        self.population_var = -1
        self.step=0

    @torch.no_grad()
    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        result = deepcopy(batch)
        data = result.access_values()
        mutation_mask = torch.rand(size=data.shape, device=data.device) < self.mutation_rate
        mutant_data = data[mutation_mask]
   
        number_idx_attack = min(mutation_mask.sum().item()//10, len(self.idx_attack))
        i_j_temp = torch.stack(
            [torch.tensor(self.idx_attack)[torch.randint(0, number_idx_attack, mutant_data.shape)],
             torch.tensor(self.control_nodes)[torch.randint(0, len(self.control_nodes), mutant_data.shape)],]).permute(1,0)
        

        mask_swap = (i_j_temp[:,0] > i_j_temp[:, 1])
        i_j_temp[mask_swap] = i_j_temp[mask_swap][:, [1, 0]]
        
        mask_equal = (i_j_temp[:,0] == i_j_temp[:, 1])
        i_j_temp[mask_equal] = torch.tensor([self.idx_attack[0], self.idx_attack[1]])
        new_pop = triu_to_linear_idx(self.n_nodes, i_j_temp[:, 0], i_j_temp[:, 1]).to(data.device)
        
        new_perm = torch.randperm(new_pop.shape[0])
        data[mutation_mask] = new_pop[new_perm]

        return result

    def update_locals(self, population, **kwargs):
        best_perturbation = self.adversary.searcher.status["pop_best"].values
        attr_adversary, adj_adversary = self.adversary._create_perturbed_graph(self.adversary.attr, self.adversary.adj, best_perturbation)
        model = self.adversary.model
        logits = model(attr_adversary, adj_adversary)
        labels = self.adversary.labels
        preds = logits.max(1)[1].type_as(labels)
        self.idx_attack = self.adversary.idx_attack[torch.where((preds[self.adversary.idx_attack] == labels[self.adversary.idx_attack]))[0].cpu()]
        self.idx_attack = self.idx_attack[torch.randperm(len(self.idx_attack))]
       
 

class IdxMutation(GraphMutation):
    def __init__(self, problem, idx_attack, n_nodes, mutation_rate=0.1, adversary=None):
        super().__init__(problem)
        self.mutation_rate = mutation_rate
        self.idx_attack =idx_attack.clone() if isinstance(idx_attack, torch.Tensor) else torch.tensor(idx_attack)
        self.n_nodes = n_nodes
        self.adversary = adversary
        self.initial_mutation_rate = mutation_rate
        self.last_variance= -1
        self.population_var = -1
        self.step=0

    @torch.no_grad()
    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        result = deepcopy(batch)
        data = result.access_values()
        mutation_mask = torch.rand(size=data.shape, device=data.device) < self.mutation_rate
        mutant_data = data[mutation_mask]

        number_idx_attack = min(20, len(self.idx_attack))
        i_j_temp = torch.stack([torch.tensor(self.idx_attack)[torch.randint(0, number_idx_attack, mutant_data.shape)],torch.randint(0, self.n_nodes, mutant_data.shape)]).permute(1,0)
        mask_swap = (i_j_temp[:,0] > i_j_temp[:, 1])
        i_j_temp[mask_swap] = i_j_temp[mask_swap][:, [1, 0]]
        
        mask_equal = (i_j_temp[:,0] == i_j_temp[:, 1])
        i_j_temp[mask_equal] = torch.tensor([self.idx_attack[0], self.idx_attack[1]])
        new_pop = triu_to_linear_idx(self.n_nodes, i_j_temp[:, 0], i_j_temp[:, 1]).to(data.device)
        
        new_perm = torch.randperm(new_pop.shape[0])
        data[mutation_mask] = new_pop[new_perm]

        return result

    def update_locals(self, population, **kwargs):
        

        best_perturbation = self.adversary.searcher.status["pop_best"].values
        attr_adversary, adj_adversary = self.adversary._create_perturbed_graph(self.adversary.attr, self.adversary.adj, best_perturbation)
        model = self.adversary.model
        logits = model(attr_adversary, adj_adversary)
        labels = self.adversary.labels
        preds = logits.max(1)[1].type_as(labels)
        self.idx_attack = self.adversary.idx_attack[torch.where((preds[self.adversary.idx_attack] == labels[self.adversary.idx_attack]))[0].cpu()]
        self.idx_attack = self.idx_attack[torch.randperm(len(self.idx_attack))]


class LocalIdxMutation(IdxMutation):
    def __init__(self, problem, idx_attack, n_nodes, mutation_rate=0.1, adversary=None):
        super().__init__(problem, idx_attack, n_nodes, mutation_rate, adversary)
        self.local_budget = self.adversary.local_budget.int()
        
    
    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        result = deepcopy(batch)
        data = result.access_values()
        self.idx_attack = torch.tensor(self.idx_attack).to(data.device)
        mutation_mask = (torch.rand(size=data.shape, device=data.device) < self.mutation_rate) | (data < 0)
        mutant_data = data[mutation_mask]

        data_rows, data_cols = linear_to_triu_idx(self.n_nodes, data)

        new_mutatnt_rows = []
        new_mutatnt_cols = []
        n_max_mutants = mutation_mask.sum(1).max()
        for i in range(data.shape[0]):
            violations, capacities = LocalBudgetMutation.mark_violations(
                perturbation_rows=data_rows[i], perturbation_cols=data_cols[i], n_nodes=self.n_nodes, local_budget=self.local_budget)
            available = (violations == 0)
            new_mutants_dest = torch.randint(0, available.sum(0), (n_max_mutants,)).to(data.device)
            
            idx_attack_available = self.idx_attack[available[self.idx_attack]]
            new_mutants_src = torch.randint(0, idx_attack_available.shape[0], (n_max_mutants,)).to(data.device)
    
             
            new_mutatnt_rows.append(idx_attack_available[new_mutants_src])
            new_mutatnt_cols.append(available.nonzero(as_tuple=True)[0][new_mutants_dest])

        new_mutatnt_rows = torch.stack(new_mutatnt_rows)
        new_mutatnt_cols = torch.stack(new_mutatnt_cols)
        
        swap_mask = new_mutatnt_rows > new_mutatnt_cols
        new_mutatnt_rows[swap_mask], new_mutatnt_cols[swap_mask] = new_mutatnt_cols[swap_mask], new_mutatnt_rows[swap_mask]

        sorted_mutation_mask = mutation_mask.long().sort(1, descending=True)[0].bool()[:, :n_max_mutants]
        data[mutation_mask] = triu_to_linear_idx(self.n_nodes, new_mutatnt_rows, new_mutatnt_cols)[sorted_mutation_mask]
        
        return result
    
    def update_locals(self, population, **kwargs):
        

        import torch_geometric
        degrees = torch_geometric.utils.degree(self.adversary.adj.indices()[0,:], num_nodes=self.n_nodes)
  
        best_perturbation = self.adversary.searcher.status["pop_best"].values
        attr_adversary, adj_adversary = self.adversary._create_perturbed_graph(self.adversary.attr, self.adversary.adj, best_perturbation)
        model = self.adversary.model
        logits = model(attr_adversary, adj_adversary)
        labels = self.adversary.labels
        preds = logits.max(1)[1].type_as(labels)
        self.idx_attack = self.adversary.idx_attack[torch.where((preds[self.adversary.idx_attack] == labels[self.adversary.idx_attack]))[0].cpu()]
        attack_degrees = degrees.cpu()[self.idx_attack]
        self.idx_attack = self.idx_attack[attack_degrees !=1]
        self.idx_attack = self.idx_attack[torch.randperm(len(self.idx_attack))]
        
    

        
        


class MarginMutation(GraphMutation):
    def __init__(self, problem, idx_attack, n_nodes, mutation_rate=0.1, adversary=None):
        super().__init__(problem)
        self.mutation_rate = mutation_rate
        self.idx_attack =idx_attack.clone()
        self.n_nodes = n_nodes
        self.adversary = adversary
    
    @torch.no_grad()
    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        result = deepcopy(batch)
        data = result.access_values()
        mutation_mask = torch.rand(size=data.shape, device=data.device) < self.mutation_rate
        mutant_data = data[mutation_mask]
        
        number_idx_attack =  len(self.idx_attack)
        labels = self.adversary.labels
        labels_start_node, start_node = temperature_sampling_from_vector(labels.cpu()[self.idx_attack], self.idx_attack, mutant_data.shape[0], temperature=1.1)
        labels_end_node, end_node = temperature_sampling_from_vector(labels.cpu()[ torch.tensor( self.adversary.idx_attack)], torch.tensor( self.adversary.idx_attack), mutant_data.shape[0], temperature=5)

        i_j_temp = torch.stack([start_node, end_node], dim=1).to(data.device)

        mask_swap = (i_j_temp[:,0] > i_j_temp[:, 1])
        i_j_temp[mask_swap] = i_j_temp[mask_swap][:, [1, 0]]

        new_pop = triu_to_linear_idx(self.n_nodes, i_j_temp[:, 0], i_j_temp[:, 1]).to(data.device)
        
        data[mutation_mask] = new_pop
        return result

    def update_locals(self, population, **kwargs):
        best_perturbation = self.adversary.searcher.status["pop_best"].values
        attr_adversary, adj_adversary = self.adversary._create_perturbed_graph(self.adversary.attr, self.adversary.adj, best_perturbation)
        model = self.adversary.model
        logits = model(attr_adversary, adj_adversary)
        labels = self.adversary.labels

        sorted = logits.argsort(-1)
        best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
        margin = (
            logits[np.arange(logits.size(0)), labels]
            - logits[np.arange(logits.size(0)), best_non_target_class]
        )
        
        margin = margin[self.adversary.idx_attack]
        # num_neg = torch.sum(margin < 0)
        # margin[margin <0] = torch.inf
        num_neg = torch.sum(margin < 0)
        margin[margin <-0.01] = torch.max(margin)
        sorted_margin_args = margin.argsort().cpu().numpy()
        self.idx_attack = torch.tensor(self.adversary.idx_attack)[sorted_margin_args][0: min(1000, len(margin)-num_neg)]

        # self.idx_attack = self.adversary.idx_attack[margin.cpu() >=0]
        # self.idx_attack = self.idx_attack[:-num_neg]
        # probabilities = (margin[margin.argsort()][:-num_neg])


class PriorMutation(GraphMutation):
    def __init__(self, problem, idx_attack, n_nodes, mutation_rate=0.1, adversary=None):
        super().__init__(problem)
        self.mutation_rate = mutation_rate
        self.adversary = adversary
        self.idx_attack = idx_attack
        self.n_nodes = n_nodes
    
    @torch.no_grad()
    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        result = deepcopy(batch)
        data = result.access_values()
        mutation_mask = torch.rand(size=data.shape, device=data.device) < self.mutation_rate
        mutant_data = data[mutation_mask]
        
        rand_index = torch.randint(0, self.adversary.modified_edge_index.shape[1], (mutant_data.shape[0],)).to(data.device)
        
        new_mut = self.adversary.modified_edge_index[:, rand_index]
        new_mut = triu_to_linear_idx(self.n_nodes, new_mut[0, :], new_mut[1, :]).to(data.device)
        data[mutation_mask] = new_mut

        return result

    def update_locals(self, population, **kwargs):
        pass





class IdxMutation2(GraphMutation):
    def __init__(self, problem, idx_attack, n_nodes, mutation_rate=0.1, adversary=None):
        super().__init__(problem)
        self.mutation_rate = mutation_rate
        self.idx_attack =idx_attack.clone() if isinstance(idx_attack, torch.Tensor) else torch.tensor(idx_attack)
        self.n_nodes = n_nodes
        self.adversary = adversary
        self.initial_mutation_rate = mutation_rate
        self.last_variance= -1
        self.population_var = -1
        self.step=0

    @torch.no_grad()
    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        result = deepcopy(batch)
        data = result.access_values()
        mutation_mask = torch.rand(size=data.shape, device=data.device) < self.mutation_rate
        mutant_data = data[mutation_mask]
        self.idx_attack = torch.tensor(self.idx_attack).to(data.device)


        number_idx_attack = min(500, len(self.idx_attack))
        start_point = torch.tensor(self.idx_attack)[torch.randint(0, number_idx_attack, mutant_data.shape)].to(data.device)
        labels = self.adversary.labels
        attack_labels = labels[start_point]
        
        temp_labels_broadcasted = labels.unsqueeze(1).repeat((1,attack_labels.shape[0]))   
        
        dist = (temp_labels_broadcasted != attack_labels).to(torch.float)+ torch.rand_like(temp_labels_broadcasted.to(torch.float)) * 1e-3
        end_point = dist.argmax(0)
        
        i_j_temp= torch.stack([start_point, end_point], dim=1).to(data.device)

        mask_swap = (i_j_temp[:,0] > i_j_temp[:, 1])
        i_j_temp[mask_swap] = i_j_temp[mask_swap][:, [1, 0]]

        new_pop = triu_to_linear_idx(self.n_nodes, i_j_temp[:, 0], i_j_temp[:, 1]).to(data.device)
        
        new_perm = torch.randperm(new_pop.shape[0])
        data[mutation_mask] = new_pop[new_perm]
        return result

    def update_locals(self, population, **kwargs):
        best_perturbation = self.adversary.searcher.status["pop_best"].values
        attr_adversary, adj_adversary = self.adversary._create_perturbed_graph(self.adversary.attr, self.adversary.adj, best_perturbation)
        model = self.adversary.model
        logits = model(attr_adversary, adj_adversary)
        labels = self.adversary.labels
        preds = logits.max(1)[1].type_as(labels)
        self.idx_attack = self.adversary.idx_attack[torch.where((preds[self.adversary.idx_attack] == labels[self.adversary.idx_attack]))[0].cpu()]
        self.idx_attack = self.idx_attack[torch.randperm(len(self.idx_attack))]
            
            
            



class IdxMutation3(GraphMutation):
    def __init__(self, problem, idx_attack, n_nodes, mutation_rate=0.1, adversary=None):
        super().__init__(problem)
        self.mutation_rate = mutation_rate
        self.idx_attack =idx_attack.clone() if isinstance(idx_attack, torch.Tensor) else torch.tensor(idx_attack)
        self.n_nodes = n_nodes
        self.adversary = adversary
        self.initial_mutation_rate = mutation_rate
        self.last_variance= -1
        self.population_var = -1
        self.step=0

    @torch.no_grad()
    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        result = deepcopy(batch)
        data = result.access_values()
        mutation_mask = torch.rand(size=data.shape, device=data.device) < self.mutation_rate
        mutant_data = data[mutation_mask]
        labels = self.adversary.labels


        number_idx_attack = min(100, len(self.idx_attack))
        
        labels_start_node, start_node = temperature_sampling_from_vector(labels.cpu()[self.idx_attack], self.idx_attack, 1000, temperature=1.1)
        labels_end_node, end_node = temperature_sampling_from_vector(labels.cpu(), torch.arange(self.n_nodes), 10000, temperature=5)
        i_j_temp = torch.stack(
            [torch.tensor(start_node)[torch.randint(0, len(start_node), mutant_data.shape)],
             torch.randint(0, self.n_nodes, mutant_data.shape)]).permute(1,0)

        mask_swap = (i_j_temp[:,0] > i_j_temp[:, 1])
        i_j_temp[mask_swap] = i_j_temp[mask_swap][:, [1, 0]]

        new_pop = triu_to_linear_idx(self.n_nodes, i_j_temp[:, 0], i_j_temp[:, 1]).to(data.device)
        
        data[mutation_mask] = new_pop

        return result