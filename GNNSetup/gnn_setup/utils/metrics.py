import torch
import numpy as np

def accuracy_from_data(model, attr, adj, labels, evaluation_mask):
    model.eval()
    with torch.no_grad():
        logits = model(attr, adj)
        preds = logits.max(1)[1].type_as(labels)
        acc = preds.eq(labels).double()
        acc = acc[evaluation_mask].mean()
    return acc.item()

def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor, split_idx: np.ndarray) -> float:
    return (logits.argmax(1)[split_idx] == labels[split_idx]).float().mean().item()