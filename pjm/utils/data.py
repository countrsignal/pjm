import dgl
import torch

from dgl import remove_self_loop
from dgl.data.utils import load_graphs

import json
from pathlib import Path
from typing import Union, Optional


__all__ = [
    "Batch",
    "Collator",
    "AF2SCN"
]


class Batch(object):
    def __init__(self, pids, seqs, folds):
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

    def process_data(self, device):
        # Batch graphs
        graphs = dgl.batch(self.folds)
        
        # Unpack features
        node_features = (graphs.ndata.pop('s'), graphs.ndata.pop('v'))
        edge_features = (graphs.edata.pop('s'), graphs.edata.pop('v'))

        # Route data to device
        sequences = self.seqs.to(device)
        graphs = graphs.to(device)
        # > Convert to FP16
        if torch.is_autocast_enabled():
            node_features = (
                node_features[0].half().to(device),
                node_features[1].half().to(device)
            )
            edge_features = (
                edge_features[0].half().to(device),
                edge_features[1].half().to(device)
            )
        else:
            node_features = (node_features[0].to(device), node_features[1].to(device))
            edge_features = (edge_features[0].to(device), edge_features[1].to(device))

        return sequences, graphs, node_features, edge_features


class Collator(object):

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, instances):
        pdb_ids, sequences, graphs = zip(*instances)
        # Tokenize sequences
        *_, sequences = self.tokenizer(list(zip(pdb_ids, sequences)))

        return Batch(pids=pdb_ids, seqs=sequences, folds=graphs)


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
        self.manifest = json.load((self.path / 'manifest.json').open('r'))[self.split]

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
