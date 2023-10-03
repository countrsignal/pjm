import dgl
import torch

from dgl import remove_self_loop
from dgl.data.utils import load_graphs

import json
from copy import deepcopy
from pathlib import Path
from typing import Union, Optional


__all__ = [
    "Router",
    "Batch",
    "Collator",
    "AF2SCN"
]


# Functions

def rdispatch(I, device):
    cop = ()
    for i, elem in enumerate(I):
        if (not isinstance(elem, tuple)) and (not isinstance(elem, list)):
            if I[i].device != device:
                cop = cop + (I[i].to(device), )
            else:
                cop = cop + (I[i], )
        else:
            cop = cop + (rdispatch(elem, device), )
    return cop


def unpack_structure_data(graphs, eps=1e-6):
    node_scalars = graphs.ndata.pop('s')
    node_vectors = graphs.ndata.pop('v')
    n_dists = torch.linalg.norm(node_vectors, ord=2, dim=-1, keepdims=True)
    node_vectors = node_vectors / (n_dists + eps)
    graphs.ndata['v_targs'] = node_vectors
    graphs.ndata['s_targs'] = n_dists.squeeze(-1)

    edge_scalars = graphs.edata.pop('s')
    edge_vectors = graphs.edata.pop('v')
    e_dists = torch.linalg.norm(edge_vectors, ord=2, dim=-1, keepdims=True)
    edge_vectors = edge_vectors / (e_dists + eps)
    graphs.edata['v_targs'] = edge_vectors
    graphs.edata['s_targs'] = e_dists.squeeze(-1)

    node_features = (node_scalars, node_vectors)
    edge_features = (edge_scalars, edge_vectors)
    return graphs, node_features, edge_features


def rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def nfeaturize(structures, rbf_channels):
    graph, node_feats, edge_feats = structures
    ns, nv = node_feats
    ns = torch.cat([ns, rbf(graph.ndata['s_targs'], D_count=rbf_channels).flatten(-2, -1)], dim=-1)
    node_feats = (ns, nv)
    return graph, node_feats, edge_feats


def efeaturize(structures, rbf_channels):
    graph, node_feats, edge_feats = structures
    src, dst = graph.edges()
    es, ev = edge_feats
    
    es = torch.cat([es, rbf((dst - src).abs(), D_count=rbf_channels)], dim=-1)
    edge_feats = (es, ev)
    return graph, node_feats, edge_feats


def generate_distogram(neighbor_distances, num_bins=64, range=(2.0, 22.0)):
    device = neighbor_distances.device
    bins = torch.linspace(range[0], range[1], num_bins).to(device)
    ca, cb = neighbor_distances.chunk(2, dim=1)
    ca = ca.flatten().contiguous()
    cb = cb.flatten().contiguous()
    ca_labels = torch.bucketize(ca, bins)
    cb_labels = torch.bucketize(cb, bins)
    del(bins)
    # Glycine does not have a Beta-Carbon
    glycine_mask = torch.eq(cb, 0.0)
    cb_labels[glycine_mask] = -100
    return ca_labels, cb_labels


# Classes

class Router(object):
    def __init__(self, packet):
        super().__init__()
        self._packet = packet
        
    def __setitem__(self, key, item):
        self._packet[key] = item

    def __getitem__(self, key):
        return self._packet[key]
    
    def reset(self):
        self._packet = {}
    
    def update(self, item):
        self._packet.update(item)
    
    def release(self, key):
        assert key in self._packet.keys()
        return self._packet.pop(key)
    
    def dispatch_data(self, key, device=None):
        assert key in self._packet.keys()
        if isinstance(self._packet[key], tuple) or isinstance(self._packet[key], list):
            if device is None:
                return self.release(key)
            else:
                return rdispatch(self.release(key), device)
        else:
            if device is None:
                return self.release(key)
            else:
                if self._packet[key].device != device:
                    return self.release(key).to(device)
                else:
                    return self.release(key)
    
    def route(self, device=None, ignore_iterables=True):
        keys = list(self._packet.keys())
        for k in keys:
            if isinstance(self._packet[k], tuple) or isinstance(self._packet[k], list):
                if ignore_iterables:
                    continue
                else:
                    if device is not None:
                        self._packet[k] = rdispatch(self._packet[k], device)
            else:   
                if device is not None:
                    if self._packet[k].device != device:
                        self._packet[k] = self._packet[k].to(device)


class Batch(object):
    def __init__(self, specs, pids, seqs, folds):
        self.specs = specs
        self.pids = pids
        self.seqs = seqs
        self.folds = folds

    def __len__(self):
        return len(self.pids)
    
    def __getitem__(self, idx):
        return self.pids[idx], self.seqs[idx, :], self.folds[idx]
    
    def __str__(self):
        return f"Batch(num_proteins={len(self.pids)})"
    
    def __repr__(self):
        return self.__str__()

    @torch.no_grad()
    def process_data(self, device, multi_modal=True):
        # Route sequence data to device
        sequences = self.seqs.to(device)

        if multi_modal:
            # Batch graphs
            graphs = dgl.batch(self.folds)
            
            # Unpack structure data
            graphs, node_features, edge_features = unpack_structure_data(
                graphs,
                **self.specs["unpack"],
            )

            # Featurize node and edge data
            graphs, node_features, edge_features = nfeaturize(
                (graphs, node_features, edge_features),
                **self.specs["nodes"],
            )
            graphs, node_features, edge_features = efeaturize(
                (graphs, node_features, edge_features),
                **self.specs["edges"],
            )
            ca_labels, cb_labels = generate_distogram(
                graphs.edata['s_targs'],
                **self.specs["distogram"],
            )
            graphs.edata['ca_labels'] = ca_labels
            graphs.edata['cb_labels'] = cb_labels

            # Route data to device
            graphs = graphs.to(device)
            node_features = rdispatch(node_features, device)
            edge_features = rdispatch(edge_features, device)

            return sequences, graphs, node_features, edge_features
        else:
            return sequences, None, None, None


class Collator(object):

    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

    def __call__(self, instances):
        pdb_ids, sequences, graphs = zip(*instances)
        # Tokenize sequences
        *_, sequences = self.tokenizer(list(zip(pdb_ids, sequences)))

        return Batch(specs=deepcopy(self.config), pids=pdb_ids, seqs=sequences, folds=graphs)


class AF2SCN(object):

    def __init__(
            self,
            split: str,
            max_len: Optional[int],
            dataset_path: Union[str, Path],
            _sequence_only_baseline: bool = False,
            _filter_by_plddt_coverage: Optional[float] = None,
            ):
        super().__init__()

        self.split = split.lower()
        self.path = Path(dataset_path).absolute()

        # Load manifest file
        manifest_json = list(self.path.glob("*.json"))
        assert len(manifest_json) == 1, f"Found {len(manifest_json)} manifest JSON files in {self.path}"
        self.manifest = json.load(manifest_json[0].open('r'))[self.split]

        # > Filter out large proteins
        self.max_len = max_len
        if max_len is not None:
            self.manifest = {k: v for k, v in self.manifest.items() if len(v['sequence']) <= max_len}
        
        # > Subset to only sequence data
        self.sequence_only = _sequence_only_baseline

        # > Filter out spaghetti proteins
        # [NOTE] Regions with pLDDT < 70 often have a ribbon-like appearance
        #        We filter out proteins with > X % of residues having a pLDDT below 70 
        if _filter_by_plddt_coverage is not None:
            plddt = self._read_plddt_json()
            filtered_manifest = {}
            for k, v in self.manifest.items():
                if "AF" in k.split("-"):
                    percent_below_70 = plddt[k]['between_50_70_pLDDT'] + plddt[k]['below_50_pLDDT']
                    if percent_below_70 <= _filter_by_plddt_coverage:
                        filtered_manifest.update({k: v})
                    else:
                        continue
                else:
                    # SideChainNet structures
                    filtered_manifest.update({k: v})
            # Update manifest
            self.manifest = filtered_manifest

        self.pdb_ids = list(self.manifest.keys())

    def __len__(self):
        return len(self.pdb_ids)

    def __getitem__(self, idx: int):
        pdb_id = self.pdb_ids[idx]
        return self.fetch(pdb_id)
    
    def _overwrite_manifest(self, new_manifest):
        self.manifest = new_manifest
        self.pdb_ids = list(self.manifest.keys())

    def _read_plddt_json(self):
        return json.load((self.path / 'pLDDT.json').open('r'))

    def fetch(self, pdb_id: str):
        sequence = self.manifest[pdb_id]['sequence']
        if self.sequence_only:
            return pdb_id, sequence, None
        else:
            partition_id, sample_index = self.manifest[pdb_id]['structure']
            partition_path = str(self.path / partition_id)
            try:
                graph, _ = load_graphs(partition_path, [int(sample_index)])
            except:
                print(f"Error loading graph for {pdb_id}")
                print(f"Partition: {partition_path}")
                print(f"Sample index: {sample_index}")
                print(f"Manifest: {self.manifest[pdb_id]}")
                raise ValueError
            graph = remove_self_loop(graph[0])
            return pdb_id, sequence, graph
