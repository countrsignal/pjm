import dgl, torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

import functools
from typing import Optional

from pjm.model._depreciated_gvp import GVPGNN
from pjm.model.decoder import MultiModalDecoder
from pjm.model.attention import Transformer, AttnLayerNorm, get_attn_mask


def standard_structure_module(
    node_in_dims,
    node_out_dims,
    edge_in_dims,
    edge_out_dims,
    num_mp_layers: int,
    num_tf_blocks: int,
    tf_dim: int,
    tf_depth: int,
    tf_heads: int,
    tf_dim_head: int,
    tf_dropout=0.1,
    **kwargs
):
    encoder = nn.ModuleList(
      [
       GVPGNN(
           node_in_dims,
           node_out_dims,
           edge_in_dims,
           edge_out_dims,
           num_mp_layers,
           **kwargs
       )
      ]
    )
    for _ in range(num_tf_blocks):
        encoder.append(
            Transformer(
                tf_dim,
                tf_depth,
                tf_heads,
                tf_dim_head,
                tf_dropout
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


class JointEmbeddings(nn.Module):

    def __init__(
      self,
      dim: int,
      alphabet_size: int,
      padding_index: int,
      sos_token_index: int,
      eos_token_index: int,
      cls_token_index: int,
      num_attn_blocks: int,
      num_structure_queries: int,
      structure_encoder: nn.ModuleList,
      contrastive_loss_weight: float = 1.,
      autoregressive_loss_weight: float = 1.,
      cross_exchange_decoding: bool = False,
      encoder_parallel_device: Optional[str] = None,
      decoder_parallel_device: Optional[str] = None,
      **kwargs
    ):
        super(JointEmbeddings, self).__init__()
        self.dim = dim
        self.pad_idx = padding_index
        self.sos_idx = sos_token_index
        self.eos_idx = eos_token_index
        self.cls_idx = cls_token_index
        self.contrastive_loss_weight = contrastive_loss_weight
        self.autoregressive_loss_weight = autoregressive_loss_weight
        # > contrastive learning temperature
        self.temperature = nn.ParameterDict({
          'temperature': nn.Parameter(torch.tensor([0.007]), requires_grad=False)
        })

        #################
        #    ENCODERS
        #################
        # Sequence Encoder
        self.embedding_layer = nn.Embedding(
            alphabet_size,
            dim,
            padding_idx=self.pad_idx
        )

        self.sequence_encoder = nn.ModuleList(
            [Transformer(dim, **kwargs) for _ in range(num_attn_blocks)]
        )
        self.sequence_cls_norm = AttnLayerNorm(dim)

        # Structure Encoder
        self.structure_encoder = structure_encoder

        #################
        #    DECODER
        #################
        self.decoder = MultiModalDecoder(
            dim=dim,
            num_layers=num_attn_blocks,
            alphabet_size=alphabet_size,
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
            tokens = attn_block(x=tokens, attn_mask=attn_mask, ar_masking=True)

        cls_tokens, tokens = tokens[:, -1], tokens[:, :-1]
        seq_embeds = self.sequence_cls_norm(cls_tokens)
        return seq_embeds, tokens

    def embed_structure(self, structures, attn_mask):
        # Graph neural network
        graph, node_feats, edge_feats = self.structure_encoder[0](*structures)
        _, *vector_dims = node_feats[1].size()
        vector_dims = functools.reduce(lambda i, j: i*j, vector_dims)
        node_feats = torch.cat(
            [node_feats[0], node_feats[1].reshape(graph.num_nodes(), vector_dims)],
            dim=-1
        )
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
        node_feats = node_feats[:, :-1, :]

        # Multi-Head attention
        for attn_layer in self.structure_encoder[1:]:
            node_feats = attn_layer(node_feats, attn_mask=attn_mask, ar_masking=False)

        return graph_feats, node_feats

    def forward(
      self,
      sequences,
      structures,
      return_embeddings=False,
      return_loss=True
      ):
        device = sequences.device

        # Decoder labels - all tokens except the <SOS>
        # > Model Parallelism (Decoder on separate device)
        if self.decoder_parallel_device is not None:
            decoder_labels = sequences[:, 1:].detach().clone().to(self.decoder_parallel_device)
        else:
            decoder_labels = sequences[:, 1:].detach().clone().to(device)

        # Decoder input - shifted tokens by removing <EOS>
        sequences = sequences[:, :-1]
        # Decoder mask
        seq_dec_mask = get_attn_mask(
            sequences, self.pad_idx, include_cls_token=False
        )

        # Add [CLS] token to sequences
        sequences = F.pad(sequences, (0, 1, 0, 0), value=self.cls_idx)
        # > Mask out pad tokens for sequence data
        seq_enc_mask = get_attn_mask(
            sequences, self.pad_idx, include_cls_token=False
        )

        # Encode sequences and structures
        seq_embs, seq_tokens = self.embed_sequence(sequences, attn_mask=seq_enc_mask)
        strc_embs, strc_tokens = self.embed_structure(structures, attn_mask=seq_dec_mask)

        if return_embeddings:
            return seq_embs, strc_embs

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
        autoregressive_loss = ce(logits, decoder_labels, ignore_index=self.pad_idx)
        autoregressive_loss = autoregressive_loss * self.autoregressive_loss_weight
        # > contrastive loss
        sim = einsum('i d, j d -> i j', seq_embs, strc_embs)
        sim = sim * self.temperature['temperature'].exp()
        contrastive_loss = InfoNCE_loss(sim)
        contrastive_loss = contrastive_loss * self.contrastive_loss_weight
        total_loss = autoregressive_loss + contrastive_loss

        return contrastive_loss, autoregressive_loss, total_loss
