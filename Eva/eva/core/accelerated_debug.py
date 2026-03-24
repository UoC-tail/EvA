import eva
import torch

from evotorch.core import Problem, SolutionBatch
from evotorch import operators as evo_ops
from evotorch.algorithms import GeneticAlgorithm
from evotorch.logging import StdOutLogger
from torch_geometric.data import Batch
from eva.core.evattack import EvAttack
from eva.utils import linear_to_triu_idx, triu_to_linear_idx
from eva.core.accelerated import EvAttackAccelerated

import numpy as np
from tqdm import tqdm

def triangular_matrix_index(M, rows):
    
    indices_all = []
    for row in (rows):
        start_index = (M * (M - 1)) // 2 - ((M - row) * (M - row - 1)) // 2
        
        if row < M - 1:
                indices = list(range(start_index, start_index + (M - row - 1)))
        else:
            indices = []
            
        indices_all.extend(indices)
        
    return indices_all




class GraphBatchProblemDebug(Problem):
    def __init__(self, attack_class, capacity=1, *args, **kwargs):
        self.capacity = capacity
        self.attack_class = attack_class
        self.h_probs = []
        
        super().__init__(*args, **kwargs)
        
    def get_probs(self, reset=True, previous_step_eval=None):
        return_val =self.h_probs.clone()
        return_val = torch.sparse_coo_tensor(
            indices=return_val.indices(),
            values=previous_step_eval - return_val.values(),
            size=return_val.size()
        )
        
        if reset:
            self.h_probs = []
        return return_val
    
    def _evaluate_batch(self, solutions: SolutionBatch):
        evaluation_result = []
        start_ind = 0
        self.h_probs = []
        while start_ind < len(solutions): # TODO: Check if it should be
            # iterating over the batches of the population
            end_ind = min(start_ind + self.capacity, len(solutions))
            batch = solutions[start_ind:end_ind]

            result = self.attack_class._create_perturbed_graph(
                attr=self.attack_class.attr, adj=self.attack_class.adj,
                perturbation=batch.values, device=self.attack_class.device, 
                return_pert=True)

            attr_batch, adj_batch, pert_matrix = result
            bulk_eval = self.attack_class.batch_evaluation(
                attr_batch=attr_batch, adj_batch=adj_batch, 
                labels_batch=self.attack_class.labels,
                model=self.attack_class.model, mask_attack=self.attack_class.mask_attack)
            
            pert_matrix = pert_matrix.coalesce()
            acc_matrix = torch.sparse_coo_tensor(
                indices=pert_matrix.indices(),
                values=bulk_eval[pert_matrix.indices()[2]],
                size=pert_matrix.size()
            ).cpu()
            self.h_probs.append(acc_matrix)            
            evaluation_result.append(bulk_eval)
            start_ind = end_ind
            
        evaluation_result = torch.cat(evaluation_result)
        self.h_probs = torch.cat(self.h_probs, dim=-1).coalesce()
        self.h_probs = torch.sparse_coo_tensor(
            indices=self.h_probs.indices(),
            values=self.h_probs.values(),
            size=self.h_probs.size()
        )
        self.h_probs = self.h_probs.sum(dim=-1).coalesce()
        self.h_probs = self.h_probs / evaluation_result.shape[-1]
        solutions.set_evals(evaluation_result)

class EvAttackAcceleratedDebug(EvAttackAccelerated):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.h_probs = []
        
    def find_optimal_perturbation(self, n_perturbations: int, **kwargs):
        evaluate_ga = lambda x: self._evaluate_sparse_perturbation(
            attr=self.attr, adj=self.adj, labels=self.labels, 
            mask_attack=self.mask_attack, perturbation=x, model=self.model, device=self.device)
        
        problem = GraphBatchProblemDebug(
            objective_sense="min", 
            objective_func=evaluate_ga,
            solution_length=n_perturbations,
            dtype=torch.int64,
            bounds=(0, (self.n_nodes * (self.n_nodes - 1)) // 2 - 1), device=self.device,
            capacity=self.capacity, attack_class=self
        )

        mutation_operator = self._define_mutation_class(problem)
        searcher = GeneticAlgorithm(
            problem, operators=[evo_ops.MultiPointCrossOver(problem, tournament_size=self.tournament_size, num_points=self.num_cross_over), 
                                mutation_operator],
            popsize=self.num_population, re_evaluate=False
        )
        logger = StdOutLogger(searcher, interval=self.stdout_interval)
        
        for i in range(self.n_steps):
            previous_step_mean = searcher.population.access_evals().reshape((-1,)).mean().cpu()
            if torch.isnan(previous_step_mean):
                previous_step_mean = torch.tensor(0.82)
            searcher.step()
            grad_track = problem.get_probs(reset=True, previous_step_eval=previous_step_mean)
            self.h_probs.append(grad_track)
            evals = searcher.population.access_evals().reshape((-1,))

            if self.debug_active:
                self.debug_info.append({"step": i, "evals": evals.clone().cpu()})
            
            mutation_operator.update_locals(searcher.population)
            
        best_perturbation = searcher.status["pop_best"].values
        return best_perturbation
    
    def _create_perturbed_graph(self, attr, adj, perturbation, return_pert=False, device=None):
        perturbation_serial = perturbation.reshape(-1, )
        perturbation_serial = perturbation_serial[perturbation_serial > 0]

        perturbation_rows, perturbation_cols = linear_to_triu_idx(self.n_nodes, perturbation_serial)

        if perturbation.ndim == 1:
            perturbation_layer_idx_count = torch.tensor([(perturbation > 0).sum().item()])
        else:        
            perturbation_layer_idx_count = (perturbation > 0).sum(dim=1)
        
        perturbation_layer_idx = torch.concat([
            torch.ones(size=(perturbation_layer_idx_count[i],), dtype=torch.long) * i 
            for i in range(perturbation_layer_idx_count.shape[0])
        ]).to(perturbation.device)
        pert_matrix = torch.sparse_coo_tensor(
            indices=torch.stack(
                [
                    torch.cat([perturbation_rows, perturbation_cols]),
                    torch.cat([perturbation_cols, perturbation_rows]),
                    torch.cat([perturbation_layer_idx, perturbation_layer_idx]), 
                ]), 
            values=torch.ones_like(torch.cat([perturbation_rows, perturbation_cols])),
            size=(self.n_nodes, self.n_nodes, perturbation_layer_idx_count.shape[0],)
        ).coalesce()

        broadcasted_adj = (torch.stack([adj] * pert_matrix.shape[-1], dim=-1)).coalesce()
        pert_adjs = ((broadcasted_adj + pert_matrix) - (broadcasted_adj * pert_matrix * 2)).coalesce()

        batch_pert_indices = pert_adjs.coalesce().indices()[:2, :] + pert_adjs.coalesce().indices()[2, :] * self.n_nodes
        pert_adjs_batch = torch.sparse_coo_tensor(
            indices=batch_pert_indices,
            values=pert_adjs.coalesce().values(),
            size=(self.n_nodes * perturbation_layer_idx_count.shape[0], self.n_nodes * perturbation_layer_idx_count.shape[0])
        ).coalesce() 
        if perturbation.ndim == 1:
            pert_attr_batch = attr 
        else:   
            pert_attr_batch = attr.repeat(perturbation_layer_idx_count.shape[0], 1)
        if return_pert:
            return pert_attr_batch, pert_adjs_batch, pert_matrix
        return pert_attr_batch, pert_adjs_batch
    

    def batch_evaluation(self, attr_batch, adj_batch, labels_batch, model, mask_attack):
        model.eval()
        pred = model(attr_batch, adj_batch).argmax(dim=-1)
        pred_batch = pred.reshape(-1, self.n_nodes)

        hits = (pred_batch == labels_batch)
        hits_masked = hits[:, mask_attack]

        accs = hits_masked.float().mean(dim=1)
        return accs

