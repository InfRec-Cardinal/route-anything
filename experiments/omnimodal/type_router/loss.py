"""Supervise task capability requirements, never candidate execution scores."""
import torch
import torch.nn.functional as F


def capability_targets(rows, dimensions, device):
    return torch.tensor([[row['capability_targets'].get(dimension['id'], 0.0)
                          for dimension in dimensions] for row in rows],
                        dtype=torch.float32, device=device)


def capability_loss(logits, targets):
    """Soft-label cross entropy; each row's requirement weights sum to one."""
    return -(targets * F.log_softmax(logits.float(), dim=-1)).sum(-1).mean()
