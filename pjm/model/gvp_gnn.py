import torch
from torch import nn
from esm.inverse_folding.util import rotate
from esm.inverse_folding.gvp_modules import GVPConv, GVPConvLayer

from .attention import Transformer


def standard_structure_module(
    node_in_dims,
    node_out_dims,
    edge_in_dims,
    num_edge_gvps: int,
    num_gvp_convs: int,
    embedding_dim: int,
    num_attn_layers: int,
    num_attn_heads: int,
    attn_head_dim: int,
    dropout=0.1,
    **kwargs
):
  encoder = nn.ModuleList(
      [
        GVPGNN(
          node_in_dims,
          node_out_dims,
          edge_in_dims,
          num_edge_gvps,
          num_gvp_convs,
          embedding_dim,
          **kwargs
        )
      ]
  )
  for _ in range(num_attn_layers):
    encoder.append(
        Transformer(
            embedding_dim=embedding_dim,
            depth=1,
            num_attn_heads=num_attn_heads,
            attn_head_dim=attn_head_dim,
            dropout=dropout,
        )
    )

  return encoder


class GVPGNN(nn.Module):
    def __init__(
        self,
        node_in_dims,
        node_out_dims,
        edge_in_dims,
        num_edge_gvps,
        num_gvp_convs,
        final_proj_dim,
        **kwargs
    ):
        super(GVPGNN, self).__init__()
        self.n_proj = GVPConv(in_dims=node_in_dims, out_dims=node_out_dims, edge_dims=edge_in_dims, n_layers=2)
        self.gvp_convs = nn.ModuleList(
            GVPConvLayer(
                node_dims=node_out_dims,
                edge_dims=edge_in_dims,
                n_edge_gvps=num_edge_gvps,
                vector_gate=True,
                **kwargs,
            ) for _ in range(num_gvp_convs)
        )
        node_out_dims = node_out_dims[0] + node_out_dims[1] * 3
        self.embed_gvp_output = nn.Sequential(
            nn.Linear(node_out_dims, final_proj_dim),
            nn.ReLU(),
            nn.Linear(final_proj_dim, final_proj_dim),
        )
    
    def forward(self, graph, node_feats, edge_feats, gvp_node_masks=None):
        edge_index = torch.stack(graph.edges(), dim=0)

        node_feats = self.n_proj(node_feats, edge_index, edge_feats)

        for gvp_conv in self.gvp_convs:
            node_feats, edge_feats = gvp_conv(
                node_feats,
                edge_index,
                edge_feats,
                autoregressive_x=None,
                node_mask=~gvp_node_masks if gvp_node_masks is not None else None,
            )
        # UN-pack node features into scaler and vector components
        scalars, vectors = node_feats
        # Get rotation invariant representation
        # > Make sure rotation matrix is on the same device as the vectors
        if vectors.device != graph.ndata['R'].device:
            R = graph.ndata['R'].to(vectors.device)
        else:
            R = graph.ndata['R']
        # node_feats = torch.einsum('ijk,ikl->ijl', R, node_feats)
        vectors = rotate(vectors, R.transpose(-2, -1)).flatten(-2, -1)
        # Merge node scaler and vector features
        node_feats = torch.cat([scalars, vectors], dim=-1)
        # Embed the output
        node_feats = self.embed_gvp_output(node_feats)
        # Zero out masked nodes
        # node_feats = node_feats.masked_fill(gvp_node_masks.unsqueeze(-1), 0.0)

        return graph, node_feats, edge_feats
