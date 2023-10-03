from math import sqrt

import torch
from torch import einsum
import torch.nn.functional as F


__all__ = [
    "_clip_loss",
    "_distogram_loss",
    "_node_reconstruction_loss",
]


def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()


def InfoNCE_loss(similarity: torch.Tensor) -> torch.Tensor:
    sequence_loss = contrastive_loss(similarity, dim=0)
    structure_loss = contrastive_loss(similarity, dim=1)
    return (sequence_loss + structure_loss) / 2.0


def _clip_loss(x, y, temperature):
    sim = einsum("i d, j d -> i j", x, y) / sqrt(x.size(-1))
    sim = sim * temperature
    return InfoNCE_loss(sim)


def _distogram_loss(logits, targets, ignore_index=-100):
    assert logits.size(0) == targets.size(0)
    # NOTE: Alphafold2 averages over nodes and neighbors
    loss = F.cross_entropy(
        logits,
        targets,
        ignore_index=ignore_index,
    )
    return loss


def _node_reconstruction_loss(preds, targets):
    pred_scalars, pred_vectors = preds
    targ_scalars, targ_vectors = targets

    scalar_pads = torch.not_equal(targ_scalars, 0.0) * torch.not_equal(targ_scalars, 1.0)
    vector_pads = ~torch.all(torch.eq(targ_vectors, 0.0), dim=-1)

    mse_scalars = (torch.square(targ_scalars - pred_scalars) * scalar_pads).sum(dim=-1)
    mse_vectors = (torch.linalg.norm((targ_vectors - pred_vectors), ord=2, dim=-1) * vector_pads).sum(dim=-1)
    return (mse_scalars + mse_vectors).mean()