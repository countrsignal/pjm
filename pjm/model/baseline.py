import functools
from typing import Optional, NewType

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

from .attention import get_attn_mask, AttnLayerNorm, Transformer
from .decoder import BaselineDecoder
from .masking import (
    get_sequence_mask,
    apply_sequence_mask,
    apply_random_token_swap,
)


Alphabet = NewType('Alphabet', object)


# Sequence only transformer
class BaselineModel(nn.Module):
    def __init__(
        self,
        dim: int,
        alphabet: Alphabet,
        num_layers: int,
        encoder_parallel_device: Optional[str] = None,
        decoder_parallel_device: Optional[str] = None,
        **kwargs
        ):
        super(BaselineModel, self).__init__()

        self.mask_idx = alphabet.mask_idx
        self.pad_idx = alphabet.padding_idx
        self.sos_idx = alphabet.sos_idx
        self.eos_idx = alphabet.eos_idx
        self.cls_idx = alphabet.cls_idx
        self.tokens_to_ignore = [self.sos_idx, self.eos_idx, self.cls_idx, self.pad_idx, self.mask_idx]
        self.tokens_to_keep = [alphabet.get_idx(aa) for aa in alphabet.all_toks if alphabet.get_idx(aa) not in self.tokens_to_ignore]

        self.encoder_parallel_device = encoder_parallel_device
        self.decoder_parallel_device = decoder_parallel_device

        self.embedding_layer = nn.Embedding(
            len(alphabet.all_toks),
            dim,
            padding_idx=self.pad_idx
        )
        self.encoder = nn.ModuleList([
            Transformer(dim, **kwargs) for _ in range(num_layers)
        ])
        self.decoder = BaselineDecoder(
            dim=dim,
            num_layers=num_layers,
            alphabet_size=len(alphabet.all_toks),
            **kwargs
        )
        self.sequence_cls_norm = AttnLayerNorm(dim)
    
    def dispatch_params(self):
        if self.encoder_parallel_device is not None:
            self.embedding_layer = self.embedding_layer.to(self.encoder_parallel_device)
            self.encoder = self.encoder.to(self.encoder_parallel_device)
            self.sequence_cls_norm = self.sequence_cls_norm.to(self.encoder_parallel_device)
        if self.decoder_parallel_device is not None:
            self.decoder = self.decoder.to(self.decoder_parallel_device)
    
    def mask_protocol(self, sequences, masking_prob, random_token_prob):

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

        # Labels mask
        # > Decoder attention mask
        seq_dec_mask = get_attn_mask(
            masked_sequences,
            self.pad_idx,
            include_cls_token=False,
        )
        # > > Select all non-masked tokens in labels to ignore in loss
        # > > > All non-asked tokens are replaced with pad_idx
        decoder_labels = decoder_labels.masked_fill(~tokens_mask, self.pad_idx)
        # > Model Parallelism (Decoder on separate device)
        if self.decoder_parallel_device is not None:
            decoder_labels = decoder_labels.to(self.decoder_parallel_device)
            seq_dec_mask = seq_dec_mask.to(self.decoder_parallel_device)
        else:
            decoder_labels = decoder_labels.to(sequences.device)
            seq_dec_mask = seq_dec_mask.to(sequences.device)

        # Sequence encoder attention mask
        # > Add [CLS] token to sequences
        masked_sequences = F.pad(masked_sequences, (0, 1, 0, 0), value=self.cls_idx)
        # > Mask out pad tokens for sequence data
        seq_enc_mask = get_attn_mask(
            masked_sequences,
            self.pad_idx,
            include_cls_token=False,
        )

        return masked_sequences, decoder_labels, seq_enc_mask, seq_dec_mask
    
    def embed_sequence(self, sequences, attn_mask):
        tokens = self.embedding_layer(sequences)
        for attn_block in self.encoder:
            tokens = attn_block(x=tokens, attn_mask=attn_mask, ar_masking=False)
        
        cls_tokens, tokens = tokens[:, -1], tokens[:, :-1]
        seq_embeds = self.sequence_cls_norm(cls_tokens)
        return seq_embeds, tokens
    
    def forward(
        self,
        sequences,
        masking_prob=0.15,
        random_token_prob=0.1,
        return_logits=False,
        ):

        with torch.no_grad():        
            masked_sequences, decoder_labels, seq_enc_mask, seq_dec_mask = self.mask_protocol(
                sequences,
                masking_prob,
                random_token_prob,
            )

        # Encode sequences and structures
        seq_embs, seq_tokens = self.embed_sequence(
            masked_sequences,
            attn_mask=seq_enc_mask,
        )
        # > Model Parallelism (Decoder on separate device)
        if self.decoder_parallel_device is not None:
            seq_embs = seq_embs.to(self.decoder_parallel_device)
            seq_tokens = seq_tokens.to(self.decoder_parallel_device)
        
        # Decoder
        # > sequence tokens are right shifted during auto-regressive training
        logits = self.decoder(
            x=seq_tokens,
            attn_mask=seq_dec_mask
        )

        if return_logits:
            return logits
        
        # Calculate losses
        ce = F.cross_entropy
        # > generative loss
        logits = rearrange(logits, 'b n c -> b c n')
        cross_entropy_loss = ce(logits, decoder_labels, ignore_index=self.pad_idx)

        return cross_entropy_loss