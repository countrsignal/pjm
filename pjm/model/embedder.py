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
    ):
    assert (model_type.startswith("mmplm") or model_type.startswith("baseline")), f"Model type {model_type} not supported."

    ### Load model checkpoint and extract sequence-encoder-only weights
    sequence_only = {}
    ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    if model_type.startswith("mmplm"):
        ### Load model config and initialize embedder
        model_args = ckpt["config"]["architectures"]["sequence"]
        for k, v in ckpt["model"].items():

            # Ignore everything other than the sequence encoder
            if not k.startswith("plm"):
                continue
            
            k = k.replace("plm.", "")

            if k.startswith("embedding_layer"):
                sequence_only[k] = v
            elif k.startswith("encoder"):
                sequence_only[k.replace("encoder", "sequence_encoder")] = v

    elif model_type.startswith("baseline"):
        ### Load model config and initialize embedder
        model_args = ckpt["config"]["architectures"]
        for k, v in ckpt["model"].items():

            if k.startswith("embedding_layer"):
                sequence_only[k] = v
            elif k.startswith("encoder"):
                sequence_only[k.replace("encoder", "sequence_encoder")] = v
            else:
                continue
    else:
        raise NotImplementedError
    del(ckpt)

    embedder = Embedder(
        alphabet=alphabet,
        **model_args,
    )

    embedder.load_state_dict(sequence_only, strict=True)
    return embedder


class Embedder(nn.Module):

    def __init__(
            self,
            embedding_dim: int,
            alphabet: Alphabet,
            num_attn_layers: int,
            **kwargs,
        ) -> None:
        super().__init__()
        self.dim = embedding_dim

        self.mask_idx = alphabet.mask_idx
        self.pad_idx = alphabet.padding_idx
        self.sos_idx = alphabet.sos_idx
        self.eos_idx = alphabet.eos_idx

        self.embedding_layer = nn.Embedding(
            len(alphabet.all_toks),
            self.dim,
            padding_idx=self.pad_idx
        )
        self.encoder = nn.ModuleList(
            [Transformer(self.dim, depth=1, **kwargs) for _ in range(num_attn_layers)]
        )
    
    def embed_sequence(self, sequences: LongTensor, attn_mask: Tensor) -> Tuple[Optional[Tensor], Tensor]:
        embs = self.embedding_layer(sequences)
        for layer in self.encoder:
            embs = layer(x=embs, attn_mask=attn_mask, ar_masking=False)
        return embs
    
    def forward(self, sequences: LongTensor) -> Tuple[Optional[Tensor], Tensor]:

        # Mask out pad tokens for sequence data
        attention_mask = get_attn_mask(
            sequences,
            self.pad_idx,
            include_cls_token=False,
        )

        # Embed sequences
        return self.embed_sequence(sequences, attention_mask)