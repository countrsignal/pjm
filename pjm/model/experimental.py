import dgl, torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

import functools
from typing import Optional, NewType

from .gvp_gnn import GVPGNN
from .decoder import MultiModalDecoder
from .attention import Transformer, AttnLayerNorm, get_attn_mask
from .masking import (
    get_sequence_mask,
    apply_sequence_mask,
    apply_random_token_swap,
    get_structure_mask,
    apply_structure_mask,
)


Alphabet = NewType('Alphabet', object)


def standard_structure_module(
    node_in_dims,
    node_out_dims,
    edge_in_dims,
    num_edge_gvps: int,
    num_gvp_convs: int,
    final_proj_dim: int,
    num_transformer_blocks: int,
    transformer_input_dim: int,
    transformer_block_depth: int,
    num_attns_heads: int,
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
           final_proj_dim,
           **kwargs
       )
      ]
  )
  for _ in range(num_transformer_blocks):
    encoder.append(
        Transformer(
            dim=transformer_input_dim,
            depth=transformer_block_depth,
            heads=num_attns_heads,
            head_dim=attn_head_dim,
            dropout=dropout,
        )
    )

  return encoder


def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()


def InfoNCE_loss(similarity: torch.Tensor) -> torch.Tensor:
    sequence_loss = contrastive_loss(similarity, dim=0)
    structure_loss = contrastive_loss(similarity, dim=1)
    return (sequence_loss + structure_loss) / 2.0


      # alphabet_size: int,
      # mask_index: int,
      # padding_index: int,
      # sos_token_index: int,
      # eos_token_index: int,
      # cls_token_index: int,


class CoCa(nn.Module):

  def __init__(
      self,
      dim: int,
      alphabet: Alphabet,
      num_transformer_blocks: int,
      structure_encoder: nn.ModuleList,
      contrastive_loss_weight: float = 1.,
      cross_entropy_loss_weight: float = 1.,
      cross_exchange_decoding: bool = False,
      encoder_parallel_device: Optional[str] = None,
      decoder_parallel_device: Optional[str] = None,
      **kwargs
  ):
    super(CoCa, self).__init__()
    self.dim = dim
    self.mask_idx = alphabet.mask_idx
    self.pad_idx = alphabet.padding_idx
    self.sos_idx = alphabet.sos_idx
    self.eos_idx = alphabet.eos_idx
    self.cls_idx = alphabet.cls_idx
    self.tokens_to_ignore = [self.sos_idx, self.eos_idx, self.cls_idx, self.pad_idx, self.mask_idx]
    self.tokens_to_keep = [alphabet.get_idx(aa) for aa in alphabet.all_toks if alphabet.get_idx(aa) not in self.tokens_to_ignore]
    self.contrastive_loss_weight = contrastive_loss_weight
    self.cross_entropy_loss_weight = cross_entropy_loss_weight
    # > contrastive learning temperature
    self.temperature = nn.ParameterDict({
      'temperature': nn.Parameter(torch.tensor([0.007]), requires_grad=False)
    })

    #################
    #    ENCODERS
    #################
    # Sequence Encoder
    self.embedding_layer = nn.Embedding(
        len(alphabet.all_toks),
        dim,
        padding_idx=self.pad_idx
    )
    self.sequence_encoder = nn.ModuleList(
        [Transformer(dim, **kwargs) for _ in range(num_transformer_blocks)]
    )
    self.sequence_cls_norm = AttnLayerNorm(dim)

    # Structure Encoder
    self.structure_encoder = structure_encoder

    #################
    #    DECODER
    #################
    self.decoder = MultiModalDecoder(
        dim=dim,
        num_layers=num_transformer_blocks,
        alphabet_size=len(alphabet.all_toks),
        cross_exchange=cross_exchange_decoding,
        **kwargs
    )

    #################
    #    PARALLEL
    #################
    self.encoder_parallel_device = encoder_parallel_device
    self.decoder_parallel_device = decoder_parallel_device
  
  def dispatch_params(self):
    if self.encoder_parallel_device is not None:
      # Send sequence encoder to device
      self.embedding_layer = self.embedding_layer.to(self.encoder_parallel_device)
      self.sequence_encoder = self.sequence_encoder.to(self.encoder_parallel_device)
      self.sequence_cls_norm = self.sequence_cls_norm.to(self.encoder_parallel_device)
      # Send structure encoder to device
      self.structure_encoder = self.structure_encoder.to(self.encoder_parallel_device)
    if self.decoder_parallel_device is not None:
      # Send decoder to device
      self.decoder = self.decoder.to(self.decoder_parallel_device)
      self.temperature.to(self.decoder_parallel_device)

  def embed_sequence(self, sequences, attn_mask):
    tokens = self.embedding_layer(sequences)
    for attn_block in self.sequence_encoder:
      tokens = attn_block(x=tokens, attn_mask=attn_mask, ar_masking=False)
    
    cls_tokens, tokens = tokens[:, -1], tokens[:, :-1]
    seq_embeds = self.sequence_cls_norm(cls_tokens)
    return seq_embeds, tokens

  def embed_structure(self, structures, attn_mask, gvp_node_masks):
    # Graph neural network
    graph, node_feats, edge_feats = self.structure_encoder[0](*structures, gvp_node_masks=gvp_node_masks)
    # _, *vector_dims = node_feats[1].size()
    # vector_dims = functools.reduce(lambda i, j: i*j, vector_dims)
    # node_feats = torch.cat(
    #     [node_feats[0], node_feats[1].reshape(graph.num_nodes(), vector_dims)],
    #     dim=-1
    # )
    del(edge_feats)

    with graph.local_scope():
      graph.ndata['h'] = node_feats
      graph_feats = dgl.mean_nodes(graph, 'h')

    # Reformat node embeddings
    with graph.local_scope():
      graph.ndata['x'] = node_feats
      graph_list = dgl.unbatch(graph)
      max_len = max([g.num_nodes() for g in graph_list])
      features = []
      for g in graph_list:
        n = g.num_nodes()
        sos = torch.LongTensor([self.sos_idx]).to(node_feats.device)
        eos = torch.LongTensor([self.eos_idx]).to(node_feats.device)
        pad = torch.LongTensor([self.pad_idx]).to(node_feats.device)
        
        pad = repeat(pad, '1 -> i', i=max_len-n)
        features.append(
            torch.cat(
                [
                    self.embedding_layer(sos),
                    g.ndata['x'],
                    self.embedding_layer(eos),
                    self.embedding_layer(pad)
                ],
                dim=0
            ).unsqueeze(0)
        )

    node_feats = torch.cat(features, dim=0)

    # Multi-Head attention
    for attn_layer in self.structure_encoder[1:]:
      node_feats = attn_layer(node_feats, attn_mask=attn_mask, ar_masking=False)

    return graph_feats, node_feats

  def forward(
      self,
      sequences,
      structures,
      masking_prob=0.15,
      random_token_prob=0.1,
      return_embeddings=False,
      return_loss=True
      ):
    
    device = sequences.device
    with torch.no_grad():

      # Get structure input features
      graph, node_feats, edge_feats = structures

      # Separate the sequences to be masks from input sequences
      masked_sequences = sequences.clone().detach()

      # Decoder labels - all tokens except the <SOS> if auto-regressive
      decoder_labels = sequences.clone().detach()

      # > Sequence Masks
      tokens_mask = get_sequence_mask(
        sequences,
        self.tokens_to_keep,
        mask_prob=(masking_prob * (1 - random_token_prob)),
      )
      # >> Apply masks to sequences
      masked_sequences = apply_sequence_mask(
        masked_sequences,
        tokens_mask,
        self.mask_idx,
      )
      # >> Random token replacement
      if random_token_prob > 0.0:
        masked_sequences = apply_random_token_swap(
          masked_sequences,
          tokens_mask,
          self.tokens_to_keep,
          random_token_prob,
        )

      # > Structure Masks
      gvp_node_masks = get_structure_mask(
        sequences,
        sequence_mask=tokens_mask,
        eos_index=self.eos_idx,
      )
      # >> Apply masks to structures
      corrupted_node_feats = apply_structure_mask(
        node_feats,
        gvp_node_masks,
      )

      # Labels mask
      # > > Select all non-masked tokens in labels to ignore in loss
      # > > > All non-asked tokens are replaced with pad_idx
      decoder_labels = decoder_labels.masked_fill(~tokens_mask, self.pad_idx)
      # > Model Parallelism (Decoder on separate device)
      if self.decoder_parallel_device is not None:
        decoder_labels = decoder_labels.to(self.decoder_parallel_device)
        # decoder_labels = sequences[:, 1:].detach().clone().to(self.decoder_parallel_device)
      else:
        decoder_labels = decoder_labels.to(device)
        # decoder_labels = sequences[:, 1:].detach().clone().to(device)
    
      # Decoder input - shifted tokens by removing <EOS>
      # sequences = sequences[:, :-1]
      
      # Decoder attention mask
      seq_dec_mask = get_attn_mask(
          masked_sequences,
          self.pad_idx,
          include_cls_token=False,
      )

      # Sequence encoder attention mask
      # > Add [CLS] token to sequences
      masked_sequences = F.pad(masked_sequences, (0, 1, 0, 0), value=self.cls_idx)
      # > Mask out pad tokens for sequence data
      seq_enc_mask = get_attn_mask(
          masked_sequences,
          self.pad_idx,
          # aux_mask=F.pad(tokens_mask, (0, 1, 0, 0), value=False),
          include_cls_token=False,
      )

    # Encode sequences and structures
    seq_embs, seq_tokens = self.embed_sequence(
      masked_sequences,
      attn_mask=seq_enc_mask,
    )
    strc_embs, strc_tokens = self.embed_structure(
      # structures,
      (graph, corrupted_node_feats, edge_feats),
      attn_mask=seq_dec_mask,
      # gvp_node_masks=gvp_node_masks,
      gvp_node_masks=None,
    )

    if return_embeddings:
      return (seq_embs, seq_tokens), (strc_embs, strc_tokens)

    # > Model Parallelism (Decoder on separate device)
    if self.decoder_parallel_device is not None:
      seq_embs = seq_embs.to(self.decoder_parallel_device)
      strc_embs = strc_embs.to(self.decoder_parallel_device)
      seq_tokens = seq_tokens.to(self.decoder_parallel_device)
      strc_tokens = strc_tokens.to(self.decoder_parallel_device)
      seq_dec_mask = seq_dec_mask.to(self.decoder_parallel_device)
    
    # Multi-modal decoder
    # > sequence tokens are right shifted during auto-regressive training
    logits = self.decoder(
        x=seq_tokens,
        y=strc_tokens,
        attn_mask=seq_dec_mask
    )

    if not return_loss:
      return logits
    
    # Calculate losses
    ce = F.cross_entropy
    # > generative loss
    logits = rearrange(logits, 'b n c -> b c n')
    cross_entropy_loss = ce(logits, decoder_labels, ignore_index=self.pad_idx)
    if self.training:
      cross_entropy_loss = cross_entropy_loss * self.cross_entropy_loss_weight
    # > contrastive loss
    sim = einsum('i d, j d -> i j', seq_embs, strc_embs)
    sim = sim * self.temperature['temperature'].exp()
    contrastive_loss = InfoNCE_loss(sim)
    if self.training:
      contrastive_loss = contrastive_loss * self.contrastive_loss_weight
    
    total_loss = cross_entropy_loss + contrastive_loss

    return contrastive_loss, cross_entropy_loss, total_loss
