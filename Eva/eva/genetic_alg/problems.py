import torch

from evotorch.core import Problem, SolutionBatch


class GraphEvalProblem(Problem):
    def __init__(self, attack_class, capacity=1, *args, **kwargs):
        self.capacity = capacity
        self.attack_class = attack_class
        super().__init__(*args, **kwargs)

    def _evaluate_batch(self, solutions: SolutionBatch):
        evaluation_result = []
        start_ind = 0
        while start_ind < len(solutions) - 1: 
            end_ind = min(start_ind + self.capacity, len(solutions))
            batch = solutions[start_ind:end_ind]
            outputs = [
            self.attack_class._create_perturbed_graph(attr=self.attack_class.attr, adj=self.attack_class.adj, 
                                          perturbation=perturbation.values, device=self.attack_class.device)
                for perturbation in batch]
            attr_instances = [output[0] for output in outputs]
            adj_instances = [output[1] for output in outputs]

            bulk_eval = self.attack_class.bulk_evaluation(
                attr_list=attr_instances, adj_list=adj_instances, 
                labels_list=[self.attack_class.labels] * len(attr_instances),
                model=self.attack_class.model, mask_attack=self.attack_class.mask_attack)
            evaluation_result.append(bulk_eval)
            start_ind = end_ind
        evaluation_result = torch.cat(evaluation_result)
        solutions.set_evals(evaluation_result)