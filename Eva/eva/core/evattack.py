import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor # TODO remove this dependency

from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch import operators as evo_ops
from evotorch.logging import StdOutLogger
from typing import Literal

from torch_geometric.utils import k_hop_subgraph


from gnn_setup.attacks.base_attack import SparseAttack 

from eva.genetic_alg.operators import PositiveIntMutation, AdaptiveIntMutation, VarinacePreservingMutation, IdxMutation, LocalIdxMutation, MarginMutation, PriorMutation, IdxMutation2, IdxMutation3, IdxControl
from eva.utils import linear_to_triu_idx

class EvAttack(SparseAttack):
    def __init__(
            self, metric=None, n_steps=1000,
            tournament_size=2, num_cross_over=30, num_population=500,
            mutation_rate=0.006, mutation_toggle_rate=0.0, 
            mutation_method: Literal["uniform", "fixed_var"]="uniform", mutation_config=None,
            training_idx=None,
            stdout_interval=1, debug_active=False,
            **kwargs):
        if metric is None:
            self.metric = self._metric

        super().__init__(**kwargs)
        # general attributes
        self.n_nodes = self.attr.shape[0]

        # self.idx_attack=idx_attack
        self.mask_attack = torch.zeros(self.n_nodes, dtype=torch.bool)
        self.mask_attack[self.idx_attack] = True

        # genetic algorithm parameters
        self.n_steps = n_steps
        self.tournament_size = tournament_size
        self.num_cross_over = num_cross_over
        self.mutation_rate = mutation_rate
        self.mutation_toggle_rate = mutation_toggle_rate
        self.num_population = num_population
        self.mutation_method=mutation_method
        self.mutation_config = mutation_config

        self.model = self.attacked_model
        self.training_idx = training_idx
        # temporary variable        
        self.best_perturbation = None

        self.stdout_interval = stdout_interval

        self.debug_active = debug_active
        self.debug_info = []
        
    def _define_mutation_class(self, problem, idx_attack=None, n_nodes=None, local=False):
        if self.mutation_method == "uniform":
            return PositiveIntMutation(problem, mutation_rate=self.mutation_rate, toggle_rate=self.mutation_toggle_rate)
        elif self.mutation_method == "adaptive":
            return AdaptiveIntMutation(problem, mutation_rate=self.mutation_rate, toggle_rate=self.mutation_toggle_rate)
        elif self.mutation_method == "fixed_var":
            return VarinacePreservingMutation(problem, mutation_rate=self.mutation_rate, **(self.mutation_config or {}))
        elif self.mutation_method == "idx_mutation" and local:
            return LocalIdxMutation(problem, idx_attack, n_nodes, mutation_rate=self.mutation_rate, adversary=self)
        elif self.mutation_method == "idx_mutation":
            return IdxMutation(problem, idx_attack, n_nodes, mutation_rate=self.mutation_rate, adversary=self)
        elif self.mutation_method == 'idx_margin':
            return MarginMutation(problem, idx_attack, n_nodes, mutation_rate=self.mutation_rate, adversary=self)
        elif self.mutation_method == 'prior_mutation':
            return PriorMutation(problem, idx_attack, n_nodes, mutation_rate=self.mutation_rate, adversary=self)
        elif self.mutation_method == 'idx_mutation2':
            return IdxMutation2(problem, idx_attack, n_nodes, mutation_rate=self.mutation_rate, adversary=self)
        elif self.mutation_method == 'idx_mutation3':
            return IdxMutation3(problem, idx_attack, n_nodes, mutation_rate=self.mutation_rate, adversary=self)
        elif self.mutation_method == 'idx_control':
            return IdxControl(problem, idx_attack, n_nodes, mutation_rate=self.mutation_rate, adversary=self)
         
    def _attack(self, n_perturbations: int, **kwargs):
        self.best_perturbation = self.find_optimal_perturbation(n_perturbations) # Attack is done in this step
        # the next step just decodes the attack to a graph
        self.attr_adversary, self.adj_adversary = self._create_perturbed_graph(self.attr, self.adj, self.best_perturbation)

    def find_optimal_perturbation(self, n_perturbations: int, **kwargs):
        evaluate_ga = lambda x: self._evaluate_sparse_perturbation(
            attr=self.attr, adj=self.adj, labels=self.labels, 
            mask_attack=self.mask_attack, perturbation=x, model=self.model, device=self.device)
        
        problem = Problem(
            "min", 
            objective_func=evaluate_ga,
            solution_length=n_perturbations,
            dtype=torch.int64,
            bounds=(0, (self.n_nodes * (self.n_nodes - 1)) // 2 - 1), device=self.device,
            num_actors=50
        )

        # mutation_class = PositiveIntMutation if self.mutation_method == "uniform" else AdaptiveIntMutation
        # mutation_operator = mutation_class(problem, mutation_rate=self.mutation_rate, toggle_rate=self.mutation_toggle_rate)
        mutation_operator = self._define_mutation_class(problem)

        searcher = GeneticAlgorithm(
            problem, operators=[evo_ops.MultiPointCrossOver(problem, tournament_size=self.tournament_size, num_points=self.num_cross_over), 
                                mutation_operator],
            popsize=self.num_population, re_evaluate=False
        )
        # searcher = GeneticAlgorithm(
        #     problem, operators=[evo_ops.MultiPointCrossOver(problem, tournament_size=self.tournament_size, num_points=self.num_cross_over), 
        #                         PositiveIntMutation(problem, mutation_rate=self.mutation_rate, toggle_rate=self.mutation_toggle_rate)],
        #     popsize=self.num_population, re_evaluate=False
        # )
        logger = StdOutLogger(searcher, interval=1)
        
        # searcher.run(self.n_steps)
        for i in range(self.n_steps):
            try:
                searcher.step()
            except Exception as e:
                print(e)
                raise e
            evals = searcher.population.access_evals().reshape((-1,))

            # if self.debug_active:
            #     self.debug_info.append({"step": i, "evals": evals.clone().cpu()})
            
            mutation_operator.update_locals(searcher.population)
            # if mutation_class == AdaptiveIntMutation:
            #     mutation_operator.set_variance(population_var)

        best_perturbation = searcher.status["pop_best"].values
        return best_perturbation             
    

    
    def _create_perturbed_graph(self, attr, adj, perturbation, device=None):
        if device is None:
            device = attr.device

        valid_perturbation = perturbation[perturbation > 0]
        perturbation_rows, perturbation_cols = linear_to_triu_idx(self.n_nodes, valid_perturbation)
        pert_matrix = torch.sparse_coo_tensor(
            indices=torch.stack([
                torch.cat([perturbation_rows, perturbation_cols]), torch.cat([perturbation_cols, perturbation_rows])]).to(device),
                values=torch.ones(perturbation_rows.shape[0] * 2).to(device),
                size=(self.n_nodes, self.n_nodes)).coalesce().to(device)
    
        adj_original = adj.clone().coalesce().to(device)
        pert_adj = ((adj_original + pert_matrix) - (adj_original * pert_matrix * 2)).coalesce()
        
        # import pdb; pdb.set_trace()
        return attr, pert_adj

    def _evaluate_sparse_perturbation(self, attr, adj, labels, mask_attack, perturbation, model, device=None):
        if device is None:
            device = attr.device

        pert_attr, pert_adj = self._create_perturbed_graph(attr, adj, perturbation)
        metric_result = self.metric(attr=pert_attr, adj=pert_adj, labels=labels, 
                                    model=model, mask_test=mask_attack, device=device)

        return metric_result

    @staticmethod
    def _metric(attr, adj, labels, model, mask_test, perturbation_adj=None, device=None):
        if device == None:
            device = attr.device
        
        model.eval()
        accuracy = (model(attr.to(device), adj.to(device)).argmax(dim=1)[mask_test] == labels[mask_test]).sum().item() / len(labels[mask_test])
        return accuracy

