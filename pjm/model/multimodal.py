from math import sqrt

import dgl
import torch
from einops import repeat
from torch import nn
import torch.nn.functional as F
from esm.inverse_folding.gvp_modules import GVPConv

from ..data import Router
from .gvp_gnn import standard_structure_module
from .attention import get_attn_mask, Transformer
from .losses import _clip_loss, _distogram_loss, _node_reconstruction_loss
from .masking import (
    get_sequence_mask,
    apply_sequence_mask,
    apply_random_token_swap,
    get_structure_mask,
    apply_structure_mask,
)


class CorruptionLayer(nn.Module):
    def __init__(self, alphabet):
        super().__init__()

        self.tokens_to_ignore = {
            "sos": alphabet.sos_idx,
            "eos": alphabet.eos_idx,
            "cls": alphabet.cls_idx,
            "mask": alphabet.mask_idx,
            "pad": alphabet.padding_idx
        }
        
        v = self.tokens_to_ignore.values()
        self.tokens_to_keep = [
            alphabet.get_idx(aa) for aa in alphabet.all_toks if alphabet.get_idx(aa) not in v
        ]
    
    def forward(
        self,
        sequences,
        structures,
        masking_prob=0.15,
        random_token_prob=0.1,
        ignore_index=-100,
    ):
        eos_index = self.tokens_to_ignore["eos"]
        mask_index = self.tokens_to_ignore["mask"]
        padding_index = self.tokens_to_ignore["pad"]
        with torch.no_grad():
            # Token-to-Nodes mask (will be useful later)
            t2n = torch.not_equal(sequences[:, 1:], eos_index) \
                * torch.not_equal(sequences[:, 1:], padding_index)
            
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
                mask_index,
            )
            # Update Decoder labels to ignore padded 
            decoder_labels = decoder_labels.masked_fill(~tokens_mask, ignore_index)
            
            # >> Random token replacement
            if random_token_prob > 0.0:
                masked_sequences = apply_random_token_swap(
                  masked_sequences,
                  tokens_mask,
                  self.tokens_to_keep,
                  random_token_prob,
                )

            graph, node_feats, edge_feats = structures
            # > Structure Masks
            node_mask = get_structure_mask(
                sequences,
                sequence_mask=tokens_mask,
                eos_index=eos_index,
            )
            # >> Apply masks to structures
            corrupted_node_feats = apply_structure_mask(
                node_feats,
                node_mask,
            )
        
        # Package everything into a dictionary
        pkg = {
            "sequences": masked_sequences,
            "tokens_mask": tokens_mask,
            "decoder_labels": decoder_labels,
            "structures": (graph, corrupted_node_feats, edge_feats),
            "node_mask": node_mask,
            "node_targets": node_feats,
            "tokens2nodes": t2n,
        }
        return Router(packet=pkg)


class DistoLayer(nn.Module):
    def __init__(self, dim, num_heads, num_classes, dropout):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim * num_heads, bias=False)
        self.proj = nn.Sequential(
            nn.Linear(num_heads, num_heads, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_heads, num_classes, bias=False),
        )
    
    def predict(self, n_i, n_j):
        N = n_i.size(0)
        # N_i shape: (N, D) -> (N, D)
        n_i = self.q(n_i)
        # N_j shape: (N, K, D) -> (N, K, H, D)
        n_j = self.k(n_j).reshape(N, -1, self.num_heads, self.dim)
        
        # Pair Representations shape: (N, K, H)
        pair_reprs = (n_i[:, None, None, :] @ n_j.transpose(-2, -1)).squeeze(-2) / sqrt(self.dim)
        # Logits shape: (N, K, H) -> (N, K, C) -> (N * K, C)
        logits = self.proj(pair_reprs).flatten(0, 1)
        return logits

    def forward(self, x, graph, targets, recon_mask, return_preds=False, ignore_index=-100):
        D = x.size(-1)
        
        # Aggregate neighbors of selected nodes
        selected_node_ids = graph.nodes()[recon_mask]
        # NOTE: We assume we are working with a KNN graph
        K = graph.num_edges() // graph.num_nodes()
        k_src, k_dst = graph.in_edges(selected_node_ids)

        # Forward Pass
        # LOGITS SHAPE: (NUM_SELECT_NODES X KNN'S, NUM_CLASSES)
        nodes_dst = x[k_dst.unique()]
        nodes_src = x[k_src].reshape(-1, K, D)
        logits = self.predict(nodes_dst, nodes_src)
        
        if return_preds:
            return logits
        
        # Compute loss
        # TARGETS SHAPE: (NUM_SELECT_NODES X KNN'S, )
        selected_edges_slice = graph.edge_ids(k_src, k_dst)
        targets = targets[selected_edges_slice]
        return _distogram_loss(logits, targets, ignore_index)
    

    class Denoiser(nn.Module):
        def __init__(
            self,
            node_in,
            node_out,
            edge_in,
            ang_channels=12,
            rbf_channels=52,
        ):
            super().__init__()
            
            self.angs = ang_channels
            self.rbfc = rbf_channels
            self.conv = GVPConv(
                in_dims=node_in,
                out_dims=node_out,
                edge_dims=edge_in,
                vector_gate=True,
            )
        
        def denoise(self, graph, node_feats, edge_feats, recon_mask, eps=1e-6):
            edge_index = torch.stack(graph.edges(), dim=0)
            scalars, vectors = self.conv(node_feats, edge_index, edge_feats)
            scalars = scalars[recon_mask]
            vectors = vectors[recon_mask]
            
            mags = torch.linalg.norm(vectors, ord=2, dim=-1).unsqueeze(-1)
            vectors = vectors / (mags + eps)
            
            cos = torch.cos(scalars[..., :self.angs])
            sin = torch.sin(scalars[..., self.angs:(self.angs*2)])
            rbf = torch.exp(scalars[..., -self.rbfc:])
            scalars = torch.cat([cos, sin, rbf], dim=-1)
            return scalars, vectors
        
        def forward(
            self,
            graph,
            node_feats,
            edge_feats,
            recon_mask,
            targets,
            return_preds=False,
            eps=1e-6,
        ):
            preds = self.denoise(graph, node_feats, edge_feats, recon_mask, eps)
            
            if return_preds:
                return preds
            
            targets = (targets[0][recon_mask], targets[1][recon_mask])
            return _node_reconstruction_loss(preds, targets)


class Decoder(nn.Module):
    def __init__(self, alphabet_size, dim, **kwargs):
        super().__init__()
        self.attn = Transformer(dim=dim, depth=1, **kwargs)
        self.logit_layer = nn.Linear(dim, alphabet_size, bias=False)
    
    def decode(self, x, attn_mask):
        x = self.attn(x, attn_mask, ar_masking=False)
        return self.logit_layer(x).permute(0, 2, 1)
    
    def forward(self, x, attn_mask, labels, return_preds=False, ignore_index=-100):
        logits = self.decode(x, attn_mask)
        
        if return_preds:
            return logits
        
        return F.cross_entropy(logits, labels, ignore_index=ignore_index)


class Denoiser(nn.Module):
    def __init__(
        self,
        node_in,
        node_out,
        edge_in,
        ang_channels,
        rbf_channels,
    ):
        super().__init__()
        
        self.angs = ang_channels
        self.rbfc = rbf_channels
        self.conv = GVPConv(
            in_dims=node_in,
            out_dims=node_out,
            edge_dims=edge_in,
            vector_gate=True,
        )
    
    def denoise(self, graph, node_feats, edge_feats, recon_mask, eps=1e-8):
        edge_index = torch.stack(graph.edges(), dim=0)
        scalars, vectors = self.conv(node_feats, edge_index, edge_feats)
        scalars = scalars[recon_mask]
        vectors = vectors[recon_mask]
        
        mags = torch.linalg.norm(vectors, ord=2, dim=-1).unsqueeze(-1)
        vectors = vectors / (mags + eps)
        
        cos = torch.cos(scalars[..., :self.angs])
        sin = torch.sin(scalars[..., self.angs:(self.angs*2)])
        rbf = torch.exp(scalars[..., -self.rbfc:])
        scalars = torch.cat([cos, sin, rbf], dim=-1)
        return scalars, vectors
    
    def forward(
        self,
        graph,
        node_feats,
        edge_feats,
        recon_mask,
        targets,
        return_preds=False,
        eps=1e-8,
    ):
        preds = self.denoise(graph, node_feats, edge_feats, recon_mask, eps)
        
        if return_preds:
            return preds
        
        targets = (targets[0][recon_mask], targets[1][recon_mask])
        return _node_reconstruction_loss(preds, targets)


class ProteinLM(nn.Module):
    def __init__(self, alphabet, embedding_dim, num_attn_layers, **kwargs):
        super().__init__()
        
        self.embedding_layer = nn.Embedding(
            len(alphabet.all_toks),
            embedding_dim,
            padding_idx=alphabet.padding_idx,
        )
        self.encoder = nn.ModuleList([
            Transformer(dim=embedding_dim, depth=1, **kwargs) for _ in range(num_attn_layers)
        ])
    
    def forward(self, x, attn_mask):
                
        h = self.embedding_layer(x)
        
        if h.device != attn_mask.device:
            attn_mask = attn_mask.to(h.device)
        
        for layer in self.encoder:
            h = layer(h, attn_mask, ar_masking=False)
        return h


class StructureEncoder(nn.Module):
    def __init__(
        self,
        node_in,
        node_out,
        edge_in,
        num_edge_convs,
        num_node_convs,
        proj_dim,
        num_attn_layers,
        num_attn_heads,
        attn_head_dim,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        
        self.ns, self.nv = node_out
        self.register_buffer("gather_mask", None, persistent=False)
        self.embedding_layer = nn.Embedding(3, proj_dim, padding_idx=2)
        self.encoder = standard_structure_module(
            node_in,
            node_out,
            edge_in,
            num_edge_convs,
            num_node_convs,
            embedding_dim=proj_dim,
            num_attn_layers=num_attn_layers,
            num_attn_heads=num_attn_heads,
            attn_head_dim=attn_head_dim,
            dropout=dropout,
            **kwargs
        )
    
    def gather_nodes(self, node_embeddings):
        if self.gather_mask is not None:
            node_embeddings = (
                node_embeddings[self.gather_mask][..., :self.ns],
                node_embeddings[self.gather_mask][..., self.ns:].view(-1, self.nv, 3)
            )
            return node_embeddings
        else:
            raise ValueError("Gather mask not initialized")
    
    def forward(self, graph, node_feats, edge_feats, attn_mask):
        graph, node_feats, edge_feats = self.encoder[0](
            graph, node_feats, edge_feats, None
        )
        
        # Attention over node embeddings
        current_dev = node_feats.device
        if current_dev != attn_mask.device:
            attn_mask = attn_mask.to(current_dev)
        # > Reformat node embeddings
        with graph.local_scope():
            graph.ndata['x'] = node_feats
            graph_list = dgl.unbatch(graph)
            lengths = graph.batch_num_nodes().tolist()
            max_len = max(lengths)
            gather_mask = torch.zeros((len(lengths), max_len + 2))
            features = []
            for idx, g in enumerate(graph_list):
                n = g.num_nodes()
                sos = torch.LongTensor([0]).to(current_dev)
                eos = torch.LongTensor([1]).to(current_dev)
                pad = torch.LongTensor([2]).to(current_dev)
                pad = repeat(pad, '1 -> i', i=max_len-n)
                
                gather_mask[idx, torch.arange(1, lengths[idx]+1)] = 1
                
                features.append(
                    torch.cat(
                        [
                            self.embedding_layer(sos),
                            g.ndata['x'],
                            self.embedding_layer(eos),
                            self.embedding_layer(pad)
                        ],
                        dim=0
                    )
                )

        gather_mask = gather_mask.bool().to(current_dev)
        self.register_buffer("gather_mask", gather_mask, persistent=False)
        node_feats = torch.stack(features, dim=0)

        # > Multi-Head attention
        for attn_layer in self.encoder[1:]:
            node_feats = attn_layer(node_feats, attn_mask=attn_mask, ar_masking=False)
        
        return graph, node_feats, edge_feats


class MMPLM(nn.Module):
    def __init__(
        self,
        config,
        alphabet,
        encoder_parallel_device=None,
        decoder_parallel_device=None,
    ):
        super().__init__()
        
        self.encoder_parallel_device = encoder_parallel_device
        self.decoder_parallel_device = decoder_parallel_device
        
        self.temperature = nn.ParameterDict({
          "temperature": nn.Parameter(torch.tensor([0.007]), requires_grad=False)
        })
        
        self.noise_layer = CorruptionLayer(alphabet)
        self.plm = ProteinLM(alphabet, **config["sequence"])
        self.structure_enc = StructureEncoder(**config["structure"])
        
        self.decoder = Decoder(alphabet_size=len(alphabet.all_toks), **config["decoder"])
        self.denoiser = Denoiser(**config["denoiser"])
        self.distogram_alphas = DistoLayer(**config["disto"])
        self.distogram_betas = DistoLayer(**config["disto"])
    
    def register_devices(self, encoder_parallel_device, decoder_parallel_device):
        self.encoder_parallel_device = encoder_parallel_device
        self.decoder_parallel_device = decoder_parallel_device
    
    def dispatch_params(self):
        assert self.encoder_parallel_device is not None, "Model Parallel is NOT enabled!"
        assert self.decoder_parallel_device is not None, "Model Parallel is NOT enabled!"
        assert self.encoder_parallel_device != self.decoder_parallel_device, "Device ID's cannot be identcal!"
        # Encoder
        self.structure_enc = self.structure_enc.to(self.encoder_parallel_device)
        # Decoders
        self.temperature = self.temperature.to(self.decoder_parallel_device)
        self.plm = self.plm.to(self.decoder_parallel_device)
        self.decoder = self.decoder.to(self.decoder_parallel_device)
        self.denoiser = self.denoiser.to(self.decoder_parallel_device)
        self.distogram_alphas = self.distogram_alphas.to(self.decoder_parallel_device)
        self.distogram_betas = self.distogram_betas.to(self.decoder_parallel_device)
    
    def forward(
        self,
        sequences,
        structures,
        masking_prob=0.15,
        random_token_prob=0.1,
        return_embs=False,
        return_preds=False,
        ignore_index=-100,
    ):
        router = self.noise_layer(
            sequences,
            structures,
            masking_prob=masking_prob,
            random_token_prob=random_token_prob,
            ignore_index=ignore_index,
        )
        router["attn_mask"] = get_attn_mask(
            router["sequences"],
            self.noise_layer.tokens_to_ignore["pad"],
            include_cls_token=False,
        )

        router["sequence_embs"] = self.plm(
            router.release("sequences"),
            router["attn_mask"],
        )
        router["structure_embs"] = self.structure_enc(
            *router.dispatch_data("structures", device=self.encoder_parallel_device),
            router["attn_mask"],
        )
        
        if return_embs:
            return router
        
        router.route(
            device=self.decoder_parallel_device,
            ignore_iterables=False,
        )
        return self.predict(router, return_preds=return_preds, ignore_index=ignore_index)
    
    def predict(self, router, return_preds=False, ignore_index=-100):
        # Upack router packet
        attn_mask = router.release("attn_mask")
        sequence_embs = router.release("sequence_embs")
        graph, node_embs, edge_embs = router.release("structure_embs")
        t2n = router.release("tokens2nodes")
        tokens_mask = router.release("tokens_mask")
        decoder_labels = router.release("decoder_labels")
        node_mask = router.release("node_mask")
        node_targets = router.release("node_targets")
        # (Sequence AND Structure) Contrastive loss
        if not return_preds:
            contrastive = _clip_loss(
                sequence_embs[tokens_mask],
                node_embs[tokens_mask],
                self.temperature["temperature"].exp(),
            )
        else:
            contrastive = None
        # (Sequence) Masked language loss
        residue_type = self.decoder(
            sequence_embs,
            attn_mask,
            decoder_labels,
            return_preds=return_preds,
            ignore_index=ignore_index,
        )
        # (Structure) Node reconstruction loss
        node_embs = self.structure_enc.gather_nodes(node_embs)
        residue_3d = self.denoiser(
            graph,
            node_embs,
            edge_embs,
            recon_mask=node_mask,
            targets=node_targets,
            return_preds=return_preds,
            eps=1e-6,
        )
        # (Sequence) Distogram losses
        # NOTE: Only channels associated with scalar channels are passed
        sequence_embs = sequence_embs[:, 1:][t2n]
        assert sequence_embs.size(0) == graph.num_nodes(), "Token-to-Nodes map failed!"
        neighboring_alphas = self.distogram_alphas(
            sequence_embs[:, :self.structure_enc.ns],
            graph,
            graph.edata["ca_labels"],
            recon_mask=node_mask,
            return_preds=return_preds,
            ignore_index=ignore_index,
        )
        neighboring_betas = self.distogram_betas(
            sequence_embs[:, :self.structure_enc.ns],
            graph,
            graph.edata["cb_labels"],
            recon_mask=node_mask,
            return_preds=return_preds,
            ignore_index=ignore_index,
        )
        return contrastive, residue_type, residue_3d, neighboring_alphas, neighboring_betas