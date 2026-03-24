from abc import ABC
import torch
import math
from typing import Literal
import warnings

import torch_geometric
from torch_geometric.data import Data as GraphData

# Defining Split Manager
class Split(ABC):
    def __init__(self, dataset):
        """
        Initializes the Splits object. The object is used to sample masks iteratively for training, validation and testing.

        Args:
            dataset (torch_geometric.data.Data): The input dataset.

        Attributes:
            device (torch.device): The device on which the data is stored.
            dataset (torch_geometric.data.Data): The input dataset.
            perm_idx (torch.Tensor): The randomly permuted indices of the dataset. The iterative steps will be based on these indices.
            perm_selected (torch.Tensor): A boolean tensor indicating which indices are selected.
            perm_class (torch.Tensor): The class labels corresponding to the permuted indices.
        """
        self.device = dataset.x.device
        self.dataset = dataset
        self.perm_idx = torch.randperm(self.dataset.x.shape[0]).to(self.device)
        self.perm_selected = torch.zeros_like(self.perm_idx).bool().to(self.device)
        self.perm_class = self.dataset.y[self.perm_idx]

    def alloc(self, budget, 
              budget_type:Literal["per_class", "overall"] = "per_class", 
              stratified=False, 
              return_cumulative=False, 
              return_mask=True):
        
        # region handling different budget types, and stratified samplings
        if budget < 1.01 and stratified:
            overall_n_nodes = self.dataset.x.shape[0]
            budget = math.floor(budget * overall_n_nodes)
            budget_type = "overall"
            # raise ValueError("Budget must be an integer when stratified is True")
        
        if budget < 1:
            budget = math.floor(budget * self.dataset.x.shape[0])
            if stratified:
                budget = math.floor(budget / (torch.max(self.dataset.y).item() + 1))
                if budget_type == "overall":
                    warnings.warn("Budget type is overall with a fraction and stratified=True. Using per_class budget instead.")
                budget_type = "per_class"

            else: # not stratified
                if budget_type == "per_class":
                    warnings.warn("Budget type is per_class but budget is not an integer. Using overall budget instead.")
                budget_type = "overall"

        else: # budget is integer
            if budget_type == "overall" and stratified:
                budget = math.floor(budget / len(torch.unique(self.dataset.y)))
                budget_type = "per_class"

            if budget_type == "per_class" and stratified == False:
                warnings.warn("Allocated per_class budget but stratified is False. Using overall budget with mutliplication instead.")
                budget = math.floor(budget * (self.dataset.y.max().item() + 1))
                budget_type = "overall"
        # endregion

        # region sampling based on stratified setting
        if stratified == False:
            selected = self.perm_idx[~self.perm_selected][:budget]
            flipping_idx = (self.perm_selected == False).nonzero(as_tuple=True)[0][:budget]
            self.perm_selected[flipping_idx] = True

        else:
            overall_selected = []
            for class_idx in torch.unique(self.perm_class):
                cls_idx = class_idx.item()
                class_pidx = self.perm_idx[(~self.perm_selected) & (self.perm_class == cls_idx)]
                class_selected = class_pidx[:min(budget, class_pidx.shape[0])]
                overall_selected.append(class_selected)
            overall_selected = torch.concat(overall_selected)
            out = torch.zeros_like(self.perm_idx).bool()
            out[overall_selected] = True
            self.perm_selected = self.perm_selected | out[self.perm_idx]
            selected = overall_selected
        # endregion

        # region returning based on the passed setting
        if return_cumulative:
            result = self.perm_idx[self.perm_selected]
        else:
            result = selected

        if return_mask == True:
            out = torch.zeros_like(self.perm_idx).bool()
            out[result] = True
            return out
        else:
            out = result
            return out
        # endregion
        
    def shuffle_free_idxs(self):
        free_idxs = self.perm_idx[~self.perm_selected]
        new_perm_unselected = torch.randperm(free_idxs.shape[0])

        # updaing perm_idx
        self.perm_idx[~self.perm_selected] = free_idxs[new_perm_unselected]

        # updating perm_class
        free_classes = self.perm_class[~self.perm_selected]
        self.perm_class[~self.perm_selected] = free_classes[new_perm_unselected]
        

# node-induced subgraph
def node_induced_subgraph(graph, mask):
    # new_edge_index = graph.edge_index.T[
    #     mask[graph.edge_index[0]] & mask[graph.edge_index[1]]
    #     ].T.clone()
    new_edge_index = torch_geometric.utils.subgraph(
        subset=mask, edge_index=graph.edge_index, relabel_nodes=True)[0]
    
    return GraphData(x=graph.x[mask], edge_index=new_edge_index, y=graph.y[mask])

def faulty_node_induced_subgraph(graph, mask):
    new_edge_index = graph.edge_index.T[
        mask[graph.edge_index[0]] & mask[graph.edge_index[1]]
        ].T.clone()
    
    return GraphData(x=graph.x, edge_index=new_edge_index, y=graph.y)



def node_induced_subgraph_arxiv(graph, mask):
    graph.cpu()
    edge_index = graph.adj_t.to_torch_sparse_coo_tensor().coalesce().indices()
    new_edge_index = edge_index.T[
        mask[edge_index[0]] & mask[edge_index[1]]
        ].T.clone()
    return GraphData(x=graph.x, edge_index=new_edge_index, y=graph.y.reshape(-1))

# edge-induced subgraph
def edge_induced_subgraph(graph, edge_mask):
    new_edge_index = graph.edge_index.T[edge_mask].T.clone()
    return GraphData(x=graph.x, edge_index=new_edge_index, y=graph.y)

# Union of two edges
def union_edge_index(graph, first, second):
    edge_index = torch_geometric.utils.sort_edge_index(torch.concat([first, second], dim=1))
    return GraphData(x=graph.x, edge_index=edge_index, y=graph.y)