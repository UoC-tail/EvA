import math
import torch

from evotorch.core import Problem, SolutionBatch
from evotorch import operators as evo_ops
from evotorch.algorithms import GeneticAlgorithm
from evotorch.logging import StdOutLogger

from eva.core.evattack import EvAttack
from eva.utils import linear_to_triu_idx, triu_to_linear_idx
import numpy as np
from tqdm import tqdm
from gnn_setup.attacks.prbcd import PRBCD
from gnn_setup.attacks.prbcd_constrained import LRBCD

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

class GraphBatchProblem(Problem):
    def __init__(self, attack_class, capacity=1, *args, **kwargs):
        self.capacity = capacity
        self.attack_class = attack_class
        super().__init__(*args, **kwargs)

    def _evaluate_batch(self, solutions: SolutionBatch):
        evaluation_result = []
        start_ind = 0
        while start_ind < len(solutions):
            end_ind = min(start_ind + self.capacity, len(solutions))
            batch = solutions[start_ind:end_ind]

            result = self.attack_class._create_perturbed_graph(
                attr=self.attack_class.attr, adj=self.attack_class.adj,
                perturbation=batch.values, device=self.attack_class.device)

            attr_batch, adj_batch = result

            bulk_eval = self.attack_class.batch_evaluation(
                attr_batch=attr_batch, adj_batch=adj_batch,
                labels_batch=self.attack_class.labels,
                model=self.attack_class.model, mask_attack=self.attack_class.mask_attack)
            evaluation_result.append(bulk_eval)
            start_ind = end_ind
        evaluation_result = torch.cat(evaluation_result)
        solutions.set_evals(evaluation_result)

class EvAttackAccelerated(EvAttack):
    def __init__(self, capacity=1, smart_init=True, **kwargs):
        super().__init__(**kwargs)
        self.searcher = None
        self.problem = None
        self.smart_init = smart_init
        self.capacity = capacity
        self.prob_edge = None
        self.modified_edge_index=None
        self.control_nodes = kwargs.get("control_nodes", None)

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
        mutation_operator = self._define_mutation_class(self.problem, idx_attack=idx_attack, n_nodes=self.n_nodes)

        self.searcher = GeneticAlgorithm(
            self.problem, operators=[evo_ops.MultiPointCrossOver(self.problem, tournament_size=self.tournament_size, num_points=self.num_cross_over),
                                mutation_operator],
            popsize=self.num_population, re_evaluate=False
        )

        if self.smart_init:
            if self.control_nodes is not None:
                number_idx_attack = self.mask_attack.sum().item()

                i_j_temp = torch.stack(
                    [torch.tensor(self.idx_attack)[torch.randint(0, number_idx_attack, self.searcher.population.values.shape)],
                     torch.tensor(self.control_nodes)[torch.randint(0, len(self.control_nodes), self.searcher.population.values.shape)]
                ]).permute(1, 2, 0)

                mask_swap = (i_j_temp[:,:,0] > i_j_temp[:, :, 1])
                i_j_temp[mask_swap] = i_j_temp[mask_swap][:, [1, 0]]
                mask_equal = (i_j_temp[:,:,0] == i_j_temp[:, :, 1])
                i_j_temp[mask_equal] = torch.tensor([idx_attack[0], idx_attack[1]])
                new_pop = triu_to_linear_idx(self.n_nodes, i_j_temp[:, :, 0], i_j_temp[:, :, 1])
                self.searcher.population.set_values(new_pop)
            else:
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
                self.searcher.population.set_values(new_pop)

        logger = StdOutLogger(self.searcher, interval=self.stdout_interval)

        for i in range(self.n_steps):
            self.searcher.step()
            evals = self.searcher.population.access_evals().reshape((-1,))

            if self.debug_active:
                self.debug_info.append({"step": i, "evals": evals.clone().cpu()})

            mutation_operator.update_locals(self.searcher.population)

        best_perturbation = self.searcher.status["pop_best"].values
        return best_perturbation

    def _create_perturbed_graph(self, attr, adj, perturbation, device=None):
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

        new_values = torch.where(pert_matrix.values() > 0, torch.tensor(1., device=attr.device), torch.tensor(0., device=pert_matrix.device))
        pert_matrix = torch.sparse_coo_tensor(indices=pert_matrix.indices(), values=new_values, size=pert_matrix.size(), device=pert_matrix.device).coalesce()

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

        return pert_attr_batch, pert_adjs_batch

    def batch_evaluation(self, attr_batch, adj_batch, labels_batch, model, mask_attack):
        model.eval()

        with torch.no_grad():
            pred = model(attr_batch, adj_batch).argmax(dim=-1)
        torch.cuda.empty_cache()
        pred_batch = pred.reshape(-1, self.n_nodes)
        del pred
        torch.cuda.empty_cache()
        hits = (pred_batch == labels_batch)
        hits_masked = hits[:, mask_attack]

        accs = hits_masked.float().mean(dim=1)
        return accs


class SparseSmoothingModel(torch.nn.Module):

    def __init__(self, model, attr):
        super().__init__()
        self.model = model
        self.attr = attr

    def forward(self, attr_idx, edge_idx, n, d):
        batch_size = n // self.attr.shape[0]
        return self.model(self.attr.repeat(batch_size, 1), (edge_idx, None))
