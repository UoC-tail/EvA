from typing import Literal
import copy
import numpy as np
import torch
import os
from torch.nn import functional as F
from torch_geometric.utils.random import erdos_renyi_graph

from eva.core.accelerated import EvAttackAccelerated, GraphBatchProblem

from evotorch.core import Problem, SolutionBatch
from evotorch import operators as evo_ops
from evotorch.algorithms import GeneticAlgorithm
from evotorch.logging import StdOutLogger
from torch_geometric.data import Batch
from eva.utils import linear_to_triu_idx
from sparse_smoothing.cert import p_lower_from_votes, max_radius_for_p_emp, joint_binary_certificate_grid, binary_certificate_grid
from gnn_setup.utils.tensors import hash_numpy_array
from eva.utils import linear_to_triu_idx
# from sparse_smoothing.utils import binary_perturb, sparse_perturb_multiple, sparse_perturb



class EvaCertAttack(EvAttackAccelerated):
    default_certificate_config = {"r_a": 1, "r_d": 3, "p_plus": 0.01, "p_minus": 0.1, "n_samples": 1000, "certificate_samples": 10000, "certificate_pre_samples": 200}
    def __init__(self, certificate_configs=None, capacity=256, lazy_artifacts_root=None, mode: Literal["acc", "ratio"]="acc", **kwargs):
        super().__init__(capacity=capacity, **kwargs)
        self.certificate_config = copy.deepcopy(self.default_certificate_config)
        self.certificate_config.update(certificate_configs or dict())
        self.lazy_artifacts_root = lazy_artifacts_root or "../data/lazy_artifacts"
        os.makedirs(self.lazy_artifacts_root, exist_ok=True)

        self._min_p = self.certificate_min_prob(self.certificate_config["r_a"], self.certificate_config["r_d"], 
                                                self.certificate_config["p_plus"], self.certificate_config["p_minus"])
        self.mode = mode
        self._resulting_stats = None
        
    @staticmethod
    def certificate_min_prob(r_a, r_d, p_plus, p_minus):
        max_val = 1.0
        min_val = 0.5

        while max_val - min_val > 1e-5:
            mid_val = (max_val + min_val) / 2
            certificate_grid, _, max_ra, max_rd = binary_certificate_grid(pf_plus=p_plus, pf_minus=p_minus, p_emps=np.array([mid_val]), progress_bar=False)
            
            if certificate_grid.shape[1] <= r_a or certificate_grid.shape[2] <= r_d:
                min_val = mid_val

            elif certificate_grid[0, r_a, r_d] > 0.5:
                max_val = mid_val
            else:
                min_val = mid_val

        print("Min prob = ", min_val)
        return min_val


    def _create_random_samples(self, adj=None, config=None, lazy=False, device=None):
        if device is None:
            device = self.device
        if config is None:
            config = copy.deepcopy(self.certificate_config)
        if adj is None:
            adj = self.adj
        
        # if lazy:
        #     adj_hash = hash_numpy_array(adj.indices().cpu().numpy())
        #     file_name = f"{adj_hash}-pPlus{config['p_plus']}-pMinus{config['p_minus']}-nSamples{config['n_samples']}.pt"
        #     file_path = os.path.join(self.lazy_artifacts_root, file_name)
        #     if os.path.exists(file_path):
        #         return torch.load(file_path)

        upper_triangle_filter = adj.indices()[0] <= adj.indices()[1]
        adj_upper_triangle = torch.sparse_coo_tensor(
            indices=adj.indices()[:, upper_triangle_filter],
            values=adj.values()[upper_triangle_filter],
            size=adj.size(),
        ).coalesce()
        adj_sampled = torch.stack([adj_upper_triangle] * config["n_samples"], dim=0).coalesce().to(device)

        # negative samples
        p_minus_vals = torch.rand(
            size=(adj_upper_triangle.indices().shape[1], config["n_samples"]), device=device)
        p_minus_vals = (p_minus_vals < config["p_minus"]).float()

        p_minus_mat = torch.sparse_coo_tensor(
            indices=adj_sampled.indices(),
            values=p_minus_vals.flatten(),
            size=adj_sampled.size(),
        ).coalesce()
        
        # positive samples
        no_plus_per_sample = torch.tensor([torch.binomial(torch.tensor((self.n_nodes * (self.n_nodes - 1)) / 2), prob=torch.tensor(config["p_plus"])) for i in range(adj_sampled.size(0))]).long()
        pluses_idx = [torch.randint(low=0, high=(self.n_nodes * (self.n_nodes - 1)) // 2 - 1, size=(no_plus_per_sample[i].item(),)) for i in range(adj_sampled.size(0))]
        pluses_matidx = [linear_to_triu_idx(self.n_nodes, pluses_idx[i]) for i in range(adj_sampled.size(0))]
        pluses_largegraph_idx = torch.stack([
            torch.concat([torch.ones_like(pluses_matidx[i][0]) * i for i in range(adj_sampled.size(0))]),
            torch.concat([pluses_matidx[i][0] for i in range(adj_sampled.size(0))]), 
            torch.concat([pluses_matidx[i][1] for i in range(adj_sampled.size(0))])
            ])
        p_plus_mat = torch.sparse_coo_tensor(
            indices=pluses_largegraph_idx,
            values=torch.ones_like(pluses_largegraph_idx[0]).float(),
            size=adj_sampled.size()
        ).coalesce().to(device)
        # 0, (self.n_nodes * (self.n_nodes - 1)) // 2 - 1

        # torch.bernoulli(torch.ones(adj_sampled.size()), p=config["p_plus"])
        # plus_idxs = (torch.rand(
        #     (config["n_samples"], adj.size()[0], adj.size()[1])) < config["p_plus"]
        #     ).nonzero().T.to(self.device)
        
        # p_plus_mat = torch.sparse_coo_tensor(
        #     indices=plus_idxs,
        #     values=torch.ones(plus_idxs.shape[1]).to(self.device),
        #     size=(config["n_samples"], adj.size()[0], adj.size()[1]),
        # ).coalesce().to(self.device) # TODO: Check if there is a better way
        # TODO it should be upper triangle

        # TODO: it should be now symmetric

        final_adj_sampled = (adj_sampled + p_plus_mat - (adj_sampled * p_plus_mat) - p_minus_mat)
        final_adj_sampled = final_adj_sampled.coalesce()

        final_adj_sampled = final_adj_sampled + final_adj_sampled.transpose(1, 2)
        final_adj_sampled = final_adj_sampled.coalesce()

        return final_adj_sampled
        
    def certify(self, attr, adj, labels, test_mask, config=None, return_stats=False):
        if config is None:
            config = copy.deepcopy(self.certificate_config)


        temp_config = copy.deepcopy(config)
        temp_config["n_samples"] = config["certificate_pre_samples"]
        config["n_samples"] = config["certificate_samples"]

        adj_presamples = self._create_random_samples(adj, temp_config, device=self.device)
        pre_probs = self.evaluate_batched_graphs(attr, adj_presamples, labels, self.model, test_mask, to_return="votes")

        adj_samples = self._create_random_samples(adj, config, device=self.device)
        probs = self.evaluate_batched_graphs(attr, adj_samples, labels, self.model, test_mask, to_return="votes")

        smooth_acc = (probs.argmax(1) == labels)[test_mask].sum() / test_mask.sum()

        p_emps = p_lower_from_votes(
            probs[test_mask].cpu(), pre_probs[test_mask].cpu(), alpha=0.005, n_samples=config["n_samples"])

        certificate_grid, _, max_ra, max_rd = binary_certificate_grid(pf_plus=config["p_plus"], pf_minus=config["p_minus"], p_emps=p_emps, progress_bar=False)

        if certificate_grid.shape[1] <= config["r_a"] or certificate_grid.shape[2] <= config["r_d"]:
            certified = torch.tensor([False] * test_mask.sum())
        else:
            certified = certificate_grid[:, config["r_a"], config["r_d"]] > 0.5 
        

        probs_pred = (probs.argmax(1) == labels)[test_mask]
        certified_acc = (torch.tensor(certified).to(self.device) * probs_pred).sum() / test_mask.sum()
        if return_stats:
            return certified, {"certified_acc": certified_acc, "smooth_acc": smooth_acc, "cert_ratio": (certified.sum() / test_mask.sum()).item()}
        return certified
        
    def return_stats(self):
        if self._resulting_stats is None:
            raise ValueError("Attack has not been run yet.")
        return copy.deepcopy(self._resulting_stats)
        
    def _attack(self, n_perturbations: int, **kwargs):
        self.clean_adj_samples = self._create_random_samples(self.adj)
        # self.clean_probs = self.evaluate_batched_graphs(self.attr, self.clean_adj_samples, self.labels, self.model, self.mask_attack)

        clean_certified, clean_stats = self.certify(self.attr, self.adj, self.labels, self.mask_attack, return_stats=True)
        print("Clean certified accuracy = ", clean_stats["certified_acc"])
        print("Clean smooth accuracy = ", clean_stats["smooth_acc"])
        print("Clean certified ratio = ", clean_stats["cert_ratio"])

        # print("Clean score = ", torch.maximum(self.clean_probs - self._min_p, torch.tensor(0)).mean())

        self.best_perturbation = self.find_optimal_perturbation(n_perturbations) # Attack is done in this step

        self.attr_adversary, self.adj_adversary = self._create_perturbed_graph(self.attr, self.adj, self.best_perturbation)
        pert_certified, pert_stats = self.certify(self.attr_adversary, self.adj_adversary, self.labels, self.mask_attack, return_stats=True)

        self._resulting_stats = {"clean": {"smooth_acc": clean_stats["smooth_acc"].item(), "certified_acc": clean_stats["certified_acc"].item(), "certified": clean_certified, "certified_ratio": clean_stats["cert_ratio"]},
                                    "perturbed": {"smooth_acc": pert_stats["smooth_acc"].item(), "certified_acc": pert_stats["certified_acc"].item(), "perturbed_certified": pert_certified, "certified_ratio": pert_stats["cert_ratio"]},} 
        print(clean_stats, pert_stats)

    def find_optimal_perturbation(self, n_perturbations: int, **kwargs):
        evaluate_ga = lambda x: self._evaluate_sparse_perturbation(
            attr=self.attr, adj=self.adj, labels=self.labels, 
            mask_attack=self.mask_attack, perturbation=x, model=self.model, device=self.device)
        
        problem = Problem(
            "min", 
            objective_func=evaluate_ga,
            solution_length=n_perturbations,
            dtype=torch.int64,
            bounds=(0, (self.n_nodes * (self.n_nodes - 1)) // 2 - 1), device=self.device
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

            if self.debug_active:
                self.debug_info.append({"step": i, "evals": evals.clone().cpu()})
            
            mutation_operator.update_locals(searcher.population)
            # if mutation_class == AdaptiveIntMutation:
            #     mutation_operator.set_variance(population_var)

        best_perturbation = searcher.status["pop_best"].values
        return best_perturbation             
    
    def _evaluate_sparse_perturbation(self, attr, adj, labels, mask_attack, perturbation, model, device=None):
        if device is None:
            device = attr.device

        pert_attr, pert_adj = self._create_perturbed_graph(attr, adj, perturbation)

        adj_diff = (pert_adj - adj).coalesce()
        adj_diff_mask = adj_diff.values() != 0
        adj_diff = torch.sparse_coo_tensor(
            indices=adj_diff.indices()[:, adj_diff_mask],
            values=adj_diff.values()[adj_diff_mask],
            size=adj_diff.size(),
        ).coalesce()
        ut_mask = adj_diff.indices()[0] <= adj_diff.indices()[1]
        ut_adj_diff = torch.sparse_coo_tensor(
            indices=adj_diff.indices()[:, ut_mask],
            values=adj_diff.values()[ut_mask],
            size=adj_diff.size()
        ).coalesce()
        adj_change_vals = ut_adj_diff.values()

        # negative sampling for the perturbations that add edges
        p_minus_mask = (adj_change_vals == 1)
        p_minus_samples = (torch.rand(size=(p_minus_mask.sum(), self.certificate_config["n_samples"])) < self.certificate_config["p_minus"]).float()

        adj_diff_minus_mat = torch.stack([torch.sparse_coo_tensor(
            indices=ut_adj_diff.indices()[:, p_minus_mask],
            values=ut_adj_diff.values()[p_minus_mask],
            size=ut_adj_diff.size(),
        ).coalesce()] * self.certificate_config["n_samples"], dim=0).coalesce()
        adj_diff_minus_mat = torch.sparse_coo_tensor(
            indices=adj_diff_minus_mat.indices(),
            values=p_minus_samples.flatten().to(self.device),
            size=adj_diff_minus_mat.size(),
        ).coalesce()
        adj_diff_minus_mat = adj_diff_minus_mat + adj_diff_minus_mat.transpose(1, 2)

        # positive sampling for the perturbations that remove edges
        p_plus_mask = (adj_change_vals == -1)
        p_plus_samples = (torch.rand(size=(p_plus_mask.sum(), self.certificate_config["n_samples"])) < self.certificate_config["p_plus"]).float()
        
        adj_diff_plus_mat = torch.stack([torch.sparse_coo_tensor(
            indices=ut_adj_diff.indices()[:, p_plus_mask],
            values=ut_adj_diff.values()[p_plus_mask],
            size=ut_adj_diff.size(),
        ).coalesce()] * self.certificate_config["n_samples"], dim=0).coalesce()
        adj_diff_plus_mat = torch.sparse_coo_tensor(
            indices=adj_diff_plus_mat.indices(),
            values=p_plus_samples.flatten().to(self.device),
            size=adj_diff_plus_mat.size(),
        ).coalesce()
        adj_diff_plus_mat = adj_diff_plus_mat + adj_diff_plus_mat.transpose(1, 2)

        adj_up = torch.stack([torch.sparse_coo_tensor(
            indices=adj_diff.indices()[:, adj_diff.values() == 1],
            values=torch.ones((adj_diff.values() == 1).sum()).to(device),
            size=adj_diff.size(),
        ).coalesce()] * self.certificate_config["n_samples"], dim=0).coalesce()
        adj_down = torch.stack([torch.sparse_coo_tensor(
            indices=adj_diff.indices()[:, adj_diff.values() == -1],
            values=torch.ones((adj_diff.values() == -1).sum()).to(device),
            size=adj_diff.size(),
        ).coalesce()] * self.certificate_config["n_samples"], dim=0).coalesce()

        no_change_samples = (self.clean_adj_samples - self.clean_adj_samples * (adj_up + adj_down)).coalesce()
        perturbed_adj_samples = (no_change_samples + (adj_up - adj_up * adj_diff_minus_mat) - (adj_down - adj_down * adj_diff_plus_mat)).coalesce()
        
        perturbed_probs = self.evaluate_batched_graphs(attr, adj_batch=perturbed_adj_samples, labels=labels, model=model, test_mask=mask_attack) 
        if self.mode == "acc":
            objective = (perturbed_probs[torch.arange(labels.shape[0]), labels][mask_attack] > self._min_p).sum().float() / mask_attack.sum()
        elif self.mode == "ratio":
            objective = (perturbed_probs.max(dim=-1).values[mask_attack] > self._min_p).sum().float() / mask_attack.sum()
        return objective

    def evaluate_batched_graphs(self, attr, adj_batch, labels, model, test_mask, to_return:Literal["probs", "votes", "top"]="probs"):
        smooth_preds = []
        start_ind = 0

        while start_ind <= adj_batch.size()[0]:
            end_ind = min(start_ind + self.capacity, adj_batch.size()[0])

            bulk_mask = (adj_batch.indices()[0] >= start_ind) & (adj_batch.indices()[0] < end_ind)
            bulk_indices = adj_batch.indices()[:, bulk_mask]
            bulk_indices[0] -= start_ind
            bulk_adj = torch.sparse_coo_tensor(
                indices=bulk_indices,
                values=adj_batch.values()[bulk_mask],
                size=(end_ind - start_ind, adj_batch.size()[1], adj_batch.size()[2])
            ).coalesce().to(self.device)

            concatenated_graph = self.concat_graphs(attr, adj_batch=bulk_adj)

            model.eval()
            pred_concatenated = model(concatenated_graph[0], concatenated_graph[1]).argmax(dim=-1).reshape(-1, self.n_nodes)
            smooth_preds.append(pred_concatenated)

            start_ind += self.capacity

        smooth_preds = torch.cat(smooth_preds, dim=0)
        if to_return == "votes":
            probs = F.one_hot(smooth_preds).float().sum(dim=0)
            return probs
        else:
            probs = F.one_hot(smooth_preds).float().mean(dim=0)
        if to_return == "probs":
            return probs
    
        top_probs = probs[torch.arange(probs.shape[0]), labels]
        return top_probs[test_mask]
            

    @staticmethod
    def concat_graphs(attr, adj_batch):
        n_repeat = adj_batch.size()[0]
        n_nodes = adj_batch.size()[1]
        attr_combined = attr.repeat(n_repeat, 1)

        # combining all adjacency matrices into a single large graph
        adj_combined = torch.sparse_coo_tensor(
            indices=adj_batch.indices()[1:] + adj_batch.indices()[0] * n_nodes,
            values=adj_batch.values(),
            size=(n_nodes * n_repeat, n_nodes * n_repeat)
        ).coalesce()
        return attr_combined, adj_combined
    

