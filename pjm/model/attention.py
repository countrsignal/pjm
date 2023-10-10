import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# for controlling freezing of parameters

def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)

def freeze_model_and_make_eval_(model):
    model.eval()
    freeze_all_layers_(model)

def get_attn_mask(text, pad_idx, include_cls_token=False):
  mask = (text != pad_idx) #.masked_fill(aux_mask, False)
  if include_cls_token:
    mask = F.pad(mask, (0, 1, 0, 0), value=True)
  mask = repeat(mask, 'b j -> b i j', i=mask.size(-1))
  return rearrange(mask, 'b i j -> b 1 i j')


# normalization
# they use layernorm without bias, something that pytorch does not offer


class AttnLayerNorm(nn.Module):
    def __init__(self, dim):
        super(AttnLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# residual


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# rotary positional embedding
# https://arxiv.org/abs/2104.09864


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GELU for gating the feedforward
# https://arxiv.org/abs/2002.05202


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, head_dim = 64, dropout = 0.1):
        super().__init__()
        inner_dim = head_dim * heads
        project_out = not (heads == 1 and head_dim == dim)
        
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.rotary_emb = RotaryEmbedding(head_dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        if project_out:
          self.to_out = nn.Sequential(
              nn.Linear(inner_dim, dim, bias=False),
              nn.ReLU(),
              nn.Linear(dim, dim, bias=False),
              nn.Dropout(dropout)
              )
        else:
          self.to_out = nn.Identity()
        
        # Layer norm BEFORE residual
        self.norm = AttnLayerNorm(dim)

        # For caching causal mask and rotary embeddings
        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)

        # For storing attention matrix
        self.attention_mat = []
    
    def get_attention_matrix(self):
       return self.attention_mat.pop()
    
    def get_mask(self, n, device):
      if self.mask is not None and self.mask.shape[-1] >= n:
        return self.mask[:n, :n]

      mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
      self.register_buffer("mask", mask, persistent=False)
      return mask

    def get_rotary_embedding(self, n, device):
      if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
        return self.pos_emb[:n]
      
      pos_emb = self.rotary_emb(n, device=device)
      self.register_buffer("pos_emb", pos_emb, persistent=False)
      return pos_emb

    def forward(self, x, attn_mask=None, ar_masking=False, store_attention=False):
      b, n, _, device, h = *x.shape, x.device, self.heads
      qkv = self.to_qkv(x).chunk(3, dim = -1)
      q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

      # rotary embeddings
      positions = self.get_rotary_embedding(n, device)
      q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

      dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

      # causal mask
      if ar_masking:
        causal_mask = self.get_mask(n, device)
        dots = dots.masked_fill(causal_mask, -torch.finfo(dots.dtype).max)

      # extra attention mask - for masking out attention from pad tokens
      if exists(attn_mask):
        dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)
      
      # Attention
      attn = torch.nn.functional.softmax(dots, dim=-1)
      if store_attention:
        self.attention_mat.append(attn.detach().cpu())

      out = einsum('b h i j, b h j d -> b h i d', attn, v)
      out = rearrange(out, 'b h n d -> b n (h d)')
      out =  self.to_out(out)
      return self.norm(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, head_dim = 64, dropout = 0.1):
        super().__init__()
        inner_dim = head_dim *  heads
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        if project_out:
          self.to_out = nn.Sequential(
              nn.Linear(inner_dim, dim, bias=False),
              nn.ReLU(),
              nn.Linear(dim, dim, bias=False),
              nn.Dropout(dropout)
              )
        else:
          self.to_out = nn.Identity()

        # Layer norm BEFORE residual
        self.norm = AttnLayerNorm(dim)

    def forward(self, x, context):
        b, n, _, h = *x.shape, self.heads

        k, v = self.to_kv(context).chunk(2, dim=-1)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        attn = torch.nn.functional.softmax(dots, dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return self.norm(out)


class Transformer(nn.Module):
  def __init__(self, embedding_dim, depth, num_attn_heads, attn_head_dim, dropout = 0.1):
    super().__init__()
    self.layers = nn.ModuleList([])
    for _ in range(depth):
      self.layers.append(
          Residual(
              Attention(
                  embedding_dim,
                  heads = num_attn_heads,
                  head_dim = attn_head_dim,
                  dropout = dropout,
                  )
              )
          )

  def forward(self, x, attn_mask, ar_masking=False, store_attention=False):

    if store_attention:
      attention_matrices = []

    for attn in self.layers:
      if store_attention:
        x = attn(x, attn_mask, ar_masking, store_attention=True)
        attention_matrices.append(attn.fn.get_attention_matrix())
      else:
        x = attn(x, attn_mask, ar_masking, store_attention=False)
    
    if store_attention:
      return x, attention_matrices
    else:
      return x