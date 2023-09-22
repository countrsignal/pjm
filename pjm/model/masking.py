import torch
import functools


__all__ = [
    "highlight_mask",
    "get_sequence_mask",
    "apply_sequence_mask",
    "apply_random_token_swap",
    "get_structure_mask",
    "apply_structure_mask",
]


def map_tokens_to_nodes(sequences, eos_index, padding_index):
    tokens2nodes = torch.not_equal(sequences[:, 1:], eos_index) * torch.not_equal(sequences[:, 1:], padding_index)
    node_types = sequences[:, 1:][tokens2nodes].detach().clone()
    return node_types, tokens2nodes


def highlight_mask(sequences, token_ids):
    mask_init = torch.full_like(sequences, False, dtype=torch.bool)
    mask = functools.reduce(lambda acc, el: acc | (sequences == el), token_ids, mask_init)
    return mask


def get_sequence_mask(sequences, tokens_to_keep, mask_prob):
    ignore_mask = highlight_mask(sequences, tokens_to_keep)
    masked_tokens = torch.rand(sequences.shape, device=sequences.device)
    masked_tokens = masked_tokens.masked_fill_(~ignore_mask, 1e3).le(mask_prob)
    return masked_tokens


def apply_sequence_mask(sequences, mask, mask_index):
    sequences = sequences.masked_fill(mask, mask_index)
    return sequences


def apply_random_token_swap(
  sequences,
  mask,
  tokens_to_keep,
  random_token_prob,
  ):

    src, dst = torch.where(mask == True)
    idx_mask = torch.zeros_like(src).float().uniform_(0, 1).le(random_token_prob)
    rand_tokens = torch.randint(
      low=min(tokens_to_keep),
      high=max(tokens_to_keep),
      size=sequences.shape,
      device=sequences.device,
    )
    while torch.any(rand_tokens[src[idx_mask], dst[idx_mask]] == sequences[src[idx_mask], dst[idx_mask]]):
      rand_tokens = torch.randint(
        low=min(tokens_to_keep),
        high=max(tokens_to_keep),
        size=sequences.shape,
        device=sequences.device,
      )
    sequences[src[idx_mask], dst[idx_mask]] = rand_tokens[src[idx_mask], dst[idx_mask]]
    return sequences


def get_structure_mask(sequences, sequence_mask, eos_index):
    """
        Here we use the mask created in `_sequence_masking` to mask out the corresponding
        amino acid features in the structure graphs.
    """
    # NOTE: The batch node features are of shape (batch_size * num_nodes, num_features)
    #       The sequence mask is of shape (batch_size, len_of_longest_protein)
    #       This means that the node features have no padding dimensions
    #       Therefore we need to exclude the padding dims from the mask tensor

    # scalars, vectors = node_features
    i, j  = torch.where(sequences == eos_index)
    node_masks = torch.cat(
        [
            sequence_mask[i[idx], 1:j[idx]] for idx in range(sequences.size(0))
        ],
        dim=-1,
    ).to(sequences.device)
    # Zero out the scalar features of masked nodes 
    # scalars = scalars.masked_fill(node_masks.unsqueeze(-1), 0.0)
    # Zero out the vector features of masked nodes
    # vectors = vectors.masked_fill(node_masks.view(-1, 1, 1), 0.0)
    return node_masks


def apply_structure_mask(node_features, structure_mask):
    scalars, vectors = node_features
    corrupted_scalars = scalars.clone().detach()
    corrupted_vectors = vectors.clone().detach()
    # Noise out the scalar features of masked nodes
    corrupted_scalars[structure_mask] = torch.randn_like(scalars[structure_mask])
    # Noise out the vector features of masked nodes
    corrupted_vectors[structure_mask] = torch.randn_like(vectors[structure_mask])
    return corrupted_scalars, corrupted_vectors