import torch


# Constants

EDGE_RBF_CHANNELS=8
NODE_ANGLE_CHANNELS=24

# Utility functions

def edge_rld(edge_vectors, edge_scalars=None):
    ca_vecs, cb_vecs = edge_vectors.split(1, dim=1)
    ca_dists = torch.linalg.norm(ca_vecs, ord=2, dim=-1, keepdim=True)
    cb_dists = torch.linalg.norm(cb_vecs, ord=2, dim=-1, keepdim=True)
    ca_vecs = ca_vecs / ca_dists
    cb_vecs = cb_vecs / cb_dists
    
    edge_unit_vectors = torch.cat([ca_vecs, cb_vecs], dim=1)
    ca_dists = ca_dists.view(-1)
    cb_dists = cb_dists.view(-1)
    
    # NOTE: Operating under the assumption that the number of
    #       RBF channels for both C-alphas & C_betas are 8
    if edge_scalars is not None:
        ca_zeros = torch.all(torch.eq(edge_scalars[:, :EDGE_RBF_CHANNELS], 0.0), dim=-1)
        cb_zeros = torch.all(torch.eq(edge_scalars[:, EDGE_RBF_CHANNELS:], 0.0), dim=-1)
        zeros_mask = ca_zeros + cb_zeros
        # Here we add an 'ignore_value' for distant neighbors
        ca_dists = ca_dists.masked_fill_(zeros_mask, value=-100.0)
        cb_dists = cb_dists.masked_fill_(zeros_mask, value=-100.0)

    return edge_unit_vectors, (ca_dists, cb_dists)


def _calc_distance_loss_(preds, targets, mask):
    # EDGE INPUT SHAPES: (Num Nodes * K-Neighbors, )
    # NODE INPUT SHAPES: (Num Nodes, Vector Channels)
    if mask is not None:
        mse_dist = (torch.square(targets - preds) * mask).sum(dim=-1)
    else:
        mse_dist = torch.square(targets - preds).sum(dim=-1)
    return mse_dist


def _calc_orientation_loss_(preds, targets, mask):
    # EDGE INPUT SHAPES: (Num Nodes * K-Neighbors, 3)
    # NODE INPUT SHAPES: (Num Nodes, Vector Channels, 3)
    if mask is not None:
        mse_vectors = (torch.linalg.norm((targets - preds), ord=2, dim=-1) * mask).sum(dim=-1)
    else:
        mse_vectors = torch.linalg.norm((targets - preds), ord=2, dim=-1).sum(dim=-1)
    return mse_vectors


def _calc_edge_loss_(preds_edges, target_edges, include_betas=False, ignore_index=-100.0):
    pred_scalars, pred_vectors = preds_edges
    target_scalars, target_vectors = target_edges

    if include_betas:
        ca_dists_pred, cb_dists_pred = pred_scalars.split(1, dim=1)
        ca_dists_target, cb_dists_target = target_scalars.split(1, dim=1)
        
        ca_unit_vecs_pred, cb_unit_vecs_pred = pred_vectors.split(1, dim=1)
        ca_unit_vecs_target, cb_unit_vecs_target = target_vectors.split(1, dim=1)

    if ignore_index is not None:
