from typing import Literal
import copy
import numpy as np
import torch
import os
from torch.nn import functional as F

from eva.core.accelerated import EvAttackAccelerated, GraphBatchProblem

from evotorch.core import Problem, SolutionBatch
from evotorch import operators as evo_ops
from evotorch.algorithms import GeneticAlgorithm
from evotorch.logging import StdOutLogger
from torch_geometric.data import Batch
from eva.utils import linear_to_triu_idx
from gnn_setup.utils.tensors import hash_numpy_array
from eva.utils import linear_to_triu_idx
from gnn_setup.conformal.core import ConformalClassifier as CP
from gnn_setup.conformal.scores import APSScore, TPSScore



class EvaConformal(EvAttackAccelerated):
    def __init__(self, unlabeled_idx, mode="coverage", **kwargs):
        super().__init__(**kwargs)
        self.unlabeled_idx = unlabeled_idx
        self.conformal = CP([TPSScore()])
        self.n_classes = (self.labels.max() + 1).item()
        self.y_true_mask = F.one_hot(self.labels, self.n_classes).bool()
        self.mode = mode

        self.broadcasted_unlabeled_idx = self.unlabeled_idx.repeat(self.capacity).reshape(self.capacity, -1)
        self.broadcasted_unlabeled_idx = (self.broadcasted_unlabeled_idx + (
            (torch.arange(self.capacity) * self.n_nodes).reshape(-1, 1).to(self.broadcasted_unlabeled_idx.device)))

    def batch_evaluation(self, attr_batch, adj_batch, labels_batch, model, mask_attack):
        # attr_bulk = torch.cat(attr_batch, dim=0)
        # adj_bulk = self.cat_block_sparse(adj_batch, block_size=self.n_nodes)
        # labels_bulk = torch.cat(labels_batch, dim=0)


        model.eval()
        logits = model(attr_batch, adj_batch)

        scores = self.conformal.get_scores_from_logits(logits)
        cal_scores = scores[self.broadcasted_unlabeled_idx.reshape(-1), self.labels[self.unlabeled_idx.repeat(self.capacity)]
                            ].reshape(self.capacity, -1).sort(dim=1).values
        
        index = self.conformal.weighted_quantile(torch.arange(self.unlabeled_idx.shape[0]).float(), 0.1).int()
        thresholds = cal_scores[:, index]

        reshaped_scores = scores.reshape(-1, self.n_nodes, self.n_classes)
        reshaped_pred_sets = (reshaped_scores > thresholds[:, None, None])
        eval_sets = reshaped_pred_sets[:, self.idx_attack]
        y_eval = self.labels[self.idx_attack]

        
        if self.mode == "coverage":
            coverages = eval_sets[:, self.y_true_mask[self.idx_attack]].float().mean(dim=1)
            return coverages
        elif self.mode == "set_size":
            set_sizes = eval_sets.sum(dim=-1).float().mean(dim=-1)
            return set_sizes * -1


        # print("I am here")  
        # self.labels[self.unlabeled_idx.repeat(self.capacity)]
    
        # pred_batch = pred.reshape(-1, self.n_nodes)
        
        # hits = (pred_batch == labels_batch)
        # hits_masked = hits[:, mask_attack]
        
        # accs = hits_masked.float().mean(dim=1)
        
        # # print("average accuracy", accs.mean())
        # return accs
    
    def make_cal_masks(self, unlabeled_idx, cal_fraction):
        broadcasted_unlabeled_idx = self.unlabeled_idx.repeat(self.capacity).reshape(self.capacity, -1)
        broadcasted_unlabeled_idx = (broadcasted_unlabeled_idx + (
            (torch.arange(self.capacity) * self.n_nodes).reshape(-1, 1).to(broadcasted_unlabeled_idx.device)))
        torch.stack([unlabeled_idx] * self.capacity)
