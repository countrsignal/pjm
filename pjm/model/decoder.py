from torch import nn

from .attention import Transformer, CrossAttention, AttnLayerNorm, Residual


class BaselineDecoder(nn.Module):
  def __init__(
      self,
      embedding_dim,
      alphabet_size,
      num_attn_layers,
      **kwargs,
      ):
    super(BaselineDecoder, self).__init__()
    self.stack = nn.ModuleList([
                Transformer(embedding_dim, depth=1, **kwargs) for _ in range(num_attn_layers)
                ])
    self.stack.append(AttnLayerNorm(embedding_dim))
    self.stack.append(nn.Linear(embedding_dim, alphabet_size, bias=False))

  def forward(self, x, attn_mask):
    for layer in self.stack[:-2]:
      x = layer(x, attn_mask, ar_masking=False)
    x = self.stack[-2](x)
    return self.stack[-1](x)


class CXDecoderBlock(nn.Module):
  def __init__(
      self,
      embedding_dim,
      depth,
      heads,
      head_dim,
      dropout
      ):
    super(CXDecoderBlock, self).__init__()
    self.layers = nn.ModuleList([
            Transformer(embedding_dim, depth, heads, head_dim, dropout),
            Transformer(embedding_dim, depth, heads, head_dim, dropout),
            Residual(CrossAttention(embedding_dim, heads, head_dim, dropout)),
            Residual(CrossAttention(embedding_dim, heads, head_dim, dropout))
    ])

  def forward(self, x, y, attn_mask):
    unimod_x, unimod_y, cross_xy, cross_yx = self.layers
    
    # Update unimodal representations
    x = unimod_x(x, attn_mask, ar_masking=False)
    y = unimod_y(y, attn_mask, ar_masking=False)

    # Update multimodal representations
    x_m = cross_xy(x, context=y)
    y_m = cross_yx(y, context=x)

    return x_m, y_m


class StandardDecoderBlock(nn.Module):
  def __init__(
      self,
      embedding_dim,
      depth,
      heads,
      head_dim,
      dropout
      ):
    super(StandardDecoderBlock, self).__init__()
    self.layers = nn.ModuleList([
            Transformer(embedding_dim, depth, heads, head_dim, dropout),
            Transformer(embedding_dim, depth, heads, head_dim, dropout),
            Residual(CrossAttention(embedding_dim, heads, head_dim, dropout))
    ])

  def forward(self, x, y, attn_mask):
    unimod_x, unimod_y, cross_xy = self.layers
    
    # Update unimodal representations
    x = unimod_x(x, attn_mask, ar_masking=False)
    if y.size(1) == x.size(1):
      y = unimod_y(y, attn_mask, ar_masking=False)
    else:
      y = unimod_y(y, attn_mask=None, ar_masking=False)

    # Update multimodal representations
    x = cross_xy(x, context=y)

    return x, y


class MultiModalDecoder(nn.Module):
  def __init__(
      self,
      dim,
      num_layers,
      alphabet_size,
      cross_exchange=False,
      **kwargs
      ):
    super(MultiModalDecoder, self).__init__()

    if not cross_exchange:
      self.stack = nn.ModuleList([
                StandardDecoderBlock(dim, **kwargs) for _ in range(num_layers)
                ])
    else:
      self.stack = nn.ModuleList([
                CXDecoderBlock(dim, **kwargs) for _ in range(num_layers)
                ])
    self.stack.append(AttnLayerNorm(dim))
    self.stack.append(nn.Linear(dim, alphabet_size, bias=False))
  
  def forward(self, x, y, attn_mask):
    for layer in self.stack[:-2]:
      x, y = layer(x, y, attn_mask)
    x = self.stack[-2](x)
    return self.stack[-1](x)
