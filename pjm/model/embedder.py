import json
from typing import NewType, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor, LongTensor

from .attention import get_attn_mask, AttnLayerNorm, Transformer


Alphabet = NewType('Alphabet', object)


def from_pretrained(
        model_type: str,
        alphabet: Alphabet,
        checkpoint_path: str,
        model_config_path: str,
    ):
    assert (model_type.startswith("pjm") or model_type.startswith("plm")), f"Model type {model_type} not supported."

    with open(model_config_path, "r") as f:
        model_args = json.load(f)

    transformer_config = {
        "depth": model_args["transformer_block_depth"],
        "heads": model_args["num_attns_heads"],
        "head_dim": model_args["attn_head_dim"],
        "dropout": model_args["dropout"],
    }
    embedder = Embedder(
        embedding_dim=model_args["embedding_dim"],
        alphabet=alphabet,
        num_transformer_blocks=model_args["num_sequence_transformer_blocks"] if model_type == "pjm" else model_args["num_transformer_blocks"],
        include_cls_norm=True if model_type == "pjm" else False,
        **transformer_config,
    )

    ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    sequence_only = {}
    for k, v in ckpt["model_state_dict"].items():
        if k.startswith("module.embedding_layer"):
            sequence_only[k.replace("module.", "")] = v
        elif k.startswith("module.sequence_encoder"):
            sequence_only[k.replace("module.", "")] = v
        elif k.startswith("module.sequence_cls_norm"):
            sequence_only[k.replace("module.", "")] = v
        else:
            continue
    del(ckpt)

    embedder.load_state_dict(sequence_only, strict=True)
    return embedder


class Embedder(nn.Module):

    def __init__(
            self,
            embedding_dim: int,
            alphabet: Alphabet,
            num_transformer_blocks: int,
            include_cls_norm: bool = False,
            **kwargs,
        ) -> None:
        super().__init__()
        self.dim = embedding_dim

        self.mask_idx = alphabet.mask_idx
        self.pad_idx = alphabet.padding_idx
        self.sos_idx = alphabet.sos_idx
        self.eos_idx = alphabet.eos_idx
        self.cls_idx = alphabet.cls_idx

        self.embedding_layer = nn.Embedding(
            len(alphabet.all_toks),
            self.dim,
            padding_idx=self.pad_idx
        )
        self.sequence_encoder = nn.ModuleList(
            [Transformer(self.dim, **kwargs) for _ in range(num_transformer_blocks)]
        )
        self.sequence_cls_norm = AttnLayerNorm(self.dim) if include_cls_norm else None
    
    def embed_sequence(self, sequences: LongTensor, attn_mask: Tensor) -> Tuple[Optional[Tensor], Tensor]:
        embs = self.embedding_layer(sequences)
        for attn_block in self.sequence_encoder:
            embs = attn_block(x=embs, attn_mask=attn_mask, ar_masking=False)
        
        if self.sequence_cls_norm is not None:
            cls_emb, embs = embs[:, -1], embs[:, :-1]
            cls_emb = self.sequence_cls_norm(cls_emb)
        else:
            cls_emb = None

        return cls_emb, embs
    
    def forward(self, sequences: LongTensor) -> Tuple[Optional[Tensor], Tensor]:
        if self.sequence_cls_norm is not None:
            # Sequence encoder attention mask
            # > Add [CLS] token to sequences
            sequences = F.pad(sequences, (0, 1, 0, 0), value=self.cls_idx)

        # Mask out pad tokens for sequence data
        attention_mask = get_attn_mask(
            sequences,
            self.pad_idx,
            include_cls_token=False,
        )

        # Embed sequences
        cls_emb, embs = self.embed_sequence(sequences, attention_mask)
        
        return cls_emb, embs