import torch
import functools
from torch import nn
import dgl.function as fn
import torch.nn.functional as F


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.
    
    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


class GVP(nn.Module):
    '''
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.
    
    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu6, torch.sigmoid), vector_gate=False):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi: 
            self.h_dim = h_dim or max(self.vi, self.vo) 
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            #nn.init.xavier_normal_(self.wh.weight)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            #nn.init.xavier_normal_(self.ws.weight)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                #nn.init.xavier_normal_(self.wv.weight)
                if self.vector_gate:
                    self.wsv = nn.Linear(self.so, self.vo)
                    #nn.init.xavier_normal_(self.wsv.weight)
        else:
            self.ws = nn.Linear(self.si, self.so)
            #nn.init.xavier_normal_(self.ws.weight)
        
        self.scalar_act, self.vector_act = activations
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`, 
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        '''
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)    
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo: 
                v = self.wv(vh) 
                v = torch.transpose(v, -1, -2)
                if self.vector_gate: 
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    gate = torch.sigmoid(gate).unsqueeze(-1)
                    v = torch.mul(v, gate)
                elif self.vector_act:
                    v_normalized = _norm_no_nan(v, axis=-1, keepdims=True)
                    v_update = self.vector_act(v_normalized)
                    v = torch.mul(v, v_update)
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3,
                                device=x[0].device)
        if self.scalar_act:
            s = self.scalar_act(s)
        
        return (s, v) if self.vo else s


class _VDropout(nn.Module):
    '''
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    '''
    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        '''
        :param x: `torch.Tensor` corresponding to vector channels
        '''
        if not self.training:
            return x
        
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=x.device)
        ).unsqueeze(-1)

        masked_fill = torch.masked_fill(x, mask == 0., 0.0)

        return torch.div(masked_fill, 1 - self.drop_rate)


class Dropout(nn.Module):
    '''
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)


class GVPLayerNorm(nn.Module):
    '''
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, dims):
        super(GVPLayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn


class GVPCell(nn.Module):
  def __init__(
      self,
      in_dims,
      out_dims,
      num_layers=3,
      activations=(F.relu6, torch.sigmoid),
      vector_gate=False,
      drop_rate=0.1
      ):
    super(GVPCell, self).__init__()

    GVP_ = functools.partial(
        GVP,
        activations=activations,
        vector_gate=vector_gate
        )
    ffns, dropouts, norms = [GVP_(in_dims, out_dims)], [Dropout(drop_rate)], [GVPLayerNorm(out_dims)]
    if num_layers > 1:
      for _ in range(num_layers - 1):
        ffns.append(GVP_(out_dims, out_dims))
        dropouts.append(Dropout(drop_rate))
        norms.append(GVPLayerNorm(out_dims))
    self.ffn = nn.ModuleList(ffns)
    self.norm = nn.ModuleList(norms)
    self.dropout = nn.ModuleList(dropouts)

  def forward(self, x):
    for ffn, norm, dropout in zip(self.ffn, self.norm, self.dropout):
        xh = ffn(x)
        xh = dropout(xh)
        x = norm(xh)
    return x


class GVPConv(nn.Module):
  def __init__(
      self,
      node_dims,
      edge_dims,
      **kwargs
      ):
    super(GVPConv, self).__init__()

    n_sc, n_vc = node_dims
    e_sc, e_vc = edge_dims
    # (?) WHY ARE WE DOUBLEING THE NODE FEATURES?! (2 * n_sc) & (2 * n_vc)
    # scalar_dims = (2 * n_sc) + e_sc
    # vector_dims = (2 * n_vc) + e_vc

    scalar_dims = n_sc + e_sc
    vector_dims = n_vc + e_vc

    self.e_gvp = GVPCell(edge_dims, edge_dims, **kwargs)
    self.n_conv = GVPCell((scalar_dims, vector_dims), node_dims, **kwargs)
  
  def message(self, edges):
    fused_scalars = torch.cat([edges.dst['s'], edges.data['s']], dim=-1)
    fused_vectors = torch.cat([edges.dst['v'], edges.data['v']], dim=-2)
    msg_s, msg_v = self.n_conv((fused_scalars, fused_vectors))
    return {'msg_s': msg_s, 'msg_v': msg_v}

  def forward(self, graph, node_feats, edge_feats):
    with graph.local_scope():

      ns, nv = node_feats
      graph.ndata['s'], graph.ndata['v'] = ns, nv

      es, ev = self.e_gvp(edge_feats)
      graph.edata['s'], graph.edata['v'] = es, ev

      # Message Passing Step
      graph.apply_edges(self.message)
      graph.update_all(fn.copy_e('msg_s', 'm'), fn.mean('m', 's_neigh'))
      graph.update_all(fn.copy_e('msg_v', 'm'), fn.sum('m', 'v_neigh'))

      s_neigh, v_neigh = graph.ndata['s_neigh'], graph.ndata['v_neigh']
      node_feats = (ns + s_neigh, nv + v_neigh)
      edge_feats = (es, ev)
    return graph, node_feats, edge_feats


class GVPGNN(nn.Module):
  def __init__(
      self,
      node_in_dims,
      node_out_dims,
      edge_in_dims,
      edge_out_dims,
      num_mp_layers,
      **kwargs
      ):
    super(GVPGNN, self).__init__()

    self.n_proj = GVPCell(
        node_in_dims,
        node_out_dims,
        num_layers=1,
        activations=(None, None),
        vector_gate=False,
        drop_rate=0.1
    )
    self.e_proj = GVPCell(
        edge_in_dims,
        edge_out_dims,
        num_layers=1,
        activations=(None, None),
        vector_gate=False,
        drop_rate=0.1
    )
    self.gconvs = nn.ModuleList([
        GVPConv(node_out_dims, edge_out_dims, **kwargs) for _ in range(num_mp_layers)
    ])

  def forward(self, graph, node_feats, edge_feats):
    node_feats = self.n_proj(node_feats)
    edge_feats = self.e_proj(edge_feats)
    for gconv in self.gconvs:
      graph, node_feats, edge_feats = gconv(graph, node_feats, edge_feats)
    return graph, node_feats, edge_feats
