import torch
from torch_geometric.utils import degree
from eva.core.accelerated import EvAttackAccelerated, GraphBatchProblem

from evotorch.core import Problem, SolutionBatch
from evotorch import operators as evo_ops
from eva.genetic_alg.operators import LocalBudgetMutation
from evotorch.algorithms import GeneticAlgorithm
from evotorch.logging import StdOutLogger
from torch_geometric.data import Batch

from eva.core.evattack import EvAttack
from eva.utils import linear_to_triu_idx, triu_to_linear_idx
# from eva.utils import triangular_matrix_index, triangular_column_indices
import numpy as np
from tqdm import tqdm

class EvaLocal(EvAttackAccelerated):
    def __init__(self, delta=0.5, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta # is the budget for each node. It is a fraction of its degree.
        self.adj = self.adj.coalesce()
        edge_index = self.adj.indices()[:, self.adj.values() != 0]
        self.orig_degrees = degree(edge_index[0, :], num_nodes=self.attr.shape[0])
        self.local_budget = self.delta * self.orig_degrees
        self.local_mutation=None
    

    def find_optimal_perturbation(self, n_perturbations: int, **kwargs):
        evaluate_ga = lambda x: self._evaluate_sparse_perturbation(
            attr=self.attr, adj=self.adj, labels=self.labels, 
            mask_attack=self.mask_attack, perturbation=x, model=self.model, device=self.device)
        
        self.problem = GraphBatchProblem(
            objective_sense="min", 
            objective_func=evaluate_ga,
            solution_length=n_perturbations,
            dtype=torch.int64,
            bounds=(0, (self.n_nodes * (self.n_nodes - 1)) // 2 - 1), device=self.device,
            capacity=self.capacity, attack_class=self
        )
        idx_attack = torch.where(self.mask_attack == 1)[0]
        mutation_operator = self._define_mutation_class(self.problem, idx_attack= idx_attack, n_nodes=self.n_nodes, local=True)
        self.local_mutation = LocalBudgetMutation(self.problem, delta=self.delta, adj=self.adj, n_nodes=self.n_nodes, idx_attack=torch.tensor(self.idx_attack), device=self.device)
        self.searcher = GeneticAlgorithm(
            self.problem, operators=[evo_ops.MultiPointCrossOver(self.problem, tournament_size=self.tournament_size, num_points=self.num_cross_over), 
                                 mutation_operator,self.local_mutation],
            popsize=self.num_population, re_evaluate=False
        )
        

        if self.smart_init:
            # rest_idx = (~self.mask_attack).nonzero().flatten().to(self.device)
            # counts_src = self.local_budget[torch.tensor(self.idx_attack)].int().to(self.device) # how much we can allocate from the source: (attacked) nodes
            # counts_dest = self.local_budget[rest_idx].int().to(self.device) # how much we can allocate to the destination: (non-attacked) nodes

            # counts_src_endidx = counts_src.cumsum(0) # cumulative sum of the counts: which shows any number below each entity belongs to that entity
            # counts_dest_endidx = counts_dest.cumsum(0) # cumulative sum of the counts: which shows any number below each entity belongs to that entity

            # sources_sumidx = torch.randint(0, counts_src.sum(), self.searcher.population.values.shape).to(self.device)
            # dest_sumidx = torch.randint(0, counts_dest.sum(), self.searcher.population.values.shape).to(self.device)

            # sources_idx = torch.tensor(self.idx_attack, device=self.device)[torch.searchsorted(counts_src_endidx, sources_sumidx)]
            # dest_idx = rest_idx[torch.searchsorted(counts_dest_endidx, dest_sumidx)]
            # swapper_idx = sources_idx > dest_idx
            # sources_idx[swapper_idx], dest_idx[swapper_idx] = dest_idx[swapper_idx], sources_idx[swapper_idx]
            # new_pop = triu_to_linear_idx(self.n_nodes, sources_idx, dest_idx)

            # new_pop_rows, new_pop_cols = linear_to_triu_idx(self.n_nodes, new_pop)

            # pert_degrees = torch.stack(
            #     [degree(new_pop_rows[i], num_nodes=self.n_nodes) for i in range(new_pop.shape[0])]
            #     ) + torch.stack(
            #         [degree(new_pop_cols[i], num_nodes=self.n_nodes) for i in range(new_pop.shape[0])]
            #     )
            # violations = (pert_degrees - self.local_budget.int())
            # self.searcher.population.set_values(new_pop)

            number_idx_attack = self.mask_attack.sum().item()

            i_j_temp = torch.stack(
                [torch.tensor(self.idx_attack)[torch.randint(0, number_idx_attack, self.searcher.population.values.shape)],
                 torch.randint(0, self.n_nodes, self.searcher.population.values.shape)
            ]).permute(1, 2, 0)
            
            mask_swap = (i_j_temp[:,:,0] > i_j_temp[:, :, 1])

            i_j_temp[mask_swap] = i_j_temp[mask_swap][:, [1, 0]]
            mask_equal = (i_j_temp[:,:,0] == i_j_temp[:, :, 1])
            i_j_temp[mask_equal] = torch.tensor([idx_attack[0], idx_attack[1]])
            new_pop = triu_to_linear_idx(self.n_nodes, i_j_temp[:, :, 0], i_j_temp[:, :, 1])
            new_pop_refined = LocalBudgetMutation(self.problem, delta=self.delta, adj=self.adj, n_nodes=self.n_nodes, device=self.device)._local_adjacency_filter(new_pop.to(self.device))
            print("Here I am")
            self.searcher.population.set_values(new_pop_refined)


        
        logger = StdOutLogger(self.searcher, interval=self.stdout_interval)
        
        for i in range(self.n_steps):
            self.searcher.step()
            evals = self.searcher.population.access_evals().reshape((-1,))
            # pop_copy = self.searcher.population.access_values().clone()
            # violations, _ = LocalBudgetMutation.mark_violations(perturbations=pop_copy, n_nodes=self.n_nodes, local_budget=self.local_budget)


            if self.debug_active:
                self.debug_info.append({"step": i, "evals": evals.clone().cpu()})
            if i > 10:
                self.local_mutation.update_locals(self.searcher.population, last_epoch=True)

            mutation_operator.update_locals(self.searcher.population)
        
        self.searcher.step()
        evals = self.searcher.population.access_evals().reshape((-1,))
         
        best_perturbation = self.searcher.status["pop_best"].values
        violations, _ = LocalBudgetMutation.mark_violations(perturbations=best_perturbation.clone(), n_nodes=self.n_nodes, local_budget=self.local_budget)
        assert violations.sum() == 0, "Violations are not zero at the end"
        return best_perturbation

    # def _local_adjacency_filter(self, perturbation, device):
    #     valid_perturbations = perturbation.clone()
    #     valid_perturbations[valid_perturbations < 0] = 0
    #     perturbation_rows, perturbation_cols = linear_to_triu_idx(self.n_nodes, valid_perturbations)

    #     while True:
    #         pert_degrees = torch.stack(
    #             [degree(perturbation_rows[i], num_nodes=self.attr.shape[0]) for i in range(perturbation.shape[0])]
    #             ) + torch.stack(
    #                 [degree(perturbation_cols[i], num_nodes=self.attr.shape[0]) for i in range(perturbation.shape[0])]
    #             )
    #         violations = (pert_degrees - self.local_budget)
    #         violations[violations < 0] = 0
    #         violations[violations > 0] = violations[violations > 0].ceil()

    #         if violations.sum() <= 0:
    #             break

    #         violations = violations.long()

    #         for i in range(perturbation.shape[0]):
    #             violation_nodes = violations[i].nonzero(as_tuple=True)[0]
    #             if violation_nodes.shape[0] == 0:
    #                 continue
    #             n_violation_nodes = violations[i][violation_nodes]
    #             coincidence = (
    #                 perturbation_rows[i] == violation_nodes.reshape(-1, 1)
    #                 ) | (perturbation_cols[i] == violation_nodes.reshape(-1, 1))

    #             removing_index = torch.cat([torch.tensor([0]).to(device), coincidence.sum(1).cumsum(0)[:-1]])
    #             # filter out
    #             population_index = coincidence.nonzero()[removing_index][:, 1]
    #             print("Here I am")
    #             perturbation_rows[i, population_index] = torch.randint(0, self.n_nodes, (population_index.shape[0],)).to(device)
    #             perturbation_cols[i, population_index] = torch.randint(0, self.n_nodes, (population_index.shape[0],)).to(device)

    #     swap_idx = perturbation_rows > perturbation_cols
    #     perturbation_rows[swap_idx], perturbation_cols[swap_idx] = perturbation_cols[swap_idx], perturbation_rows[swap_idx]
        
    #     new_perturbation = triu_to_linear_idx(self.n_nodes, perturbation_rows, perturbation_cols)
    #     return new_perturbationƒ