from typing import Optional, Union, Tuple, Dict
from multiprocessing import current_process
from abc import ABC, abstractmethod
from inspect import getfullargspec
import logging
import sys

from torch import nn

from .data import  AF2SCN
from .tokenizer import Alphabet
from ..model.experimental import standard_structure_module, CoCa


def _setup_logger(logger_name: str, log_level: int):
    logger = logging.getLogger(name=logger_name)
    logger.setLevel(log_level)

    # create console handler and set level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def load_jem(
    devices: Tuple[Union[int, str]],
    alphabet: Alphabet,
    model_args: Dict[str, Union[int, str, float, bool]],
    _sanity_check_mode_: bool = False,
    ) -> nn.Module:
    
    dev0, dev1 = devices
    config = {
        "dim": model_args["embedding_dim"],
        "alphabet": alphabet,
        "num_transformer_blocks": model_args["num_sequence_transformer_blocks"],
        "contrastive_loss_weight": model_args["contrastive_loss_weight"],
        "cross_entropy_loss_weight": model_args["cross_entropy_loss_weight"],
        "cross_exchange_decoding": model_args["cross_exchange_decoding"],
        "corrupt_structure": model_args["corrupt_structure"],
        "structure_reconstruction": model_args["structure_reconstruction"],
        "sturcture_global_projection": model_args["sturcture_global_projection"],
        'encoder_parallel_device':dev0,
        'decoder_parallel_device': dev1,
        'depth': model_args['transformer_block_depth'],
        'heads': model_args['num_attns_heads'],
        'head_dim': model_args['attn_head_dim'],
        'dropout': model_args["dropout"],
    }

    # Structure encoder args
    if not _sanity_check_mode_:
        strc_enc_args = {}
        strc_enc_args = {
            "node_in_dims": model_args["node_in_dims"],
            "node_out_dims": model_args["node_out_dims"],
            "edge_in_dims": model_args["edge_in_dims"],
            "num_edge_gvps": model_args["num_edge_gvps"],
            "num_gvp_convs": model_args["num_gvp_convs"],
            "final_proj_dim": model_args["embedding_dim"],
            "num_transformer_blocks": model_args["num_structure_transformer_blocks"],
            "transformer_input_dim": model_args["embedding_dim"],
            "transformer_block_depth": model_args["transformer_block_depth"],
            "num_attns_heads": model_args["num_attns_heads"],
            "attn_head_dim": model_args["attn_head_dim"],
            "dropout": model_args["dropout"],
        }
        config['structure_encoder'] = standard_structure_module(**strc_enc_args)
    else:
        config['structure_encoder'] = None

    return CoCa(**config)


def load_model(
    devices: Tuple[Union[int, str]],
    alphabet: Alphabet,
    model_args: Dict[str, Union[int, str, float, bool, nn.Module]],
    ) -> nn.Module:
    dev0, dev1 = devices
    config = {
        'encoder_parallel_device':dev0,
        'decoder_parallel_device': dev1,
        'depth': model_args['depth'],
        'heads': model_args['heads'],
        'dim_head': model_args['dim_head'],
        'dropout': 0.1
    }

    strc_enc_args = {}
    for argument in getfullargspec(standard_structure_module).args:
        strc_enc_args[argument] = model_args[argument]
    config['structure_encoder'] = standard_structure_module(**strc_enc_args)

    # config['alphabet_size'] = len(alphabet.all_toks)
    # config['padding_index'] = alphabet.padding_idx

    if model_args["model_type"].lower() == "coca":
        for argument in getfullargspec(CoCa.__init__).args[1:-2]:
            # if argument not in ('structure_encoder', 'alphabet_size', 'padding_index'):
            if argument != 'structure_encoder':
                # if argument == 'mask_index':
                #     config[argument] = alphabet.mask_idx
                # elif argument == 'sos_token_index':
                #     config[argument] = alphabet.sos_idx
                # elif argument == 'eos_token_index':
                #     config[argument] = alphabet.eos_idx
                # elif argument == 'cls_token_index':
                #     config[argument] = alphabet.cls_idx
                if argument == 'alphabet':
                    config[argument] = alphabet
                else:
                    config[argument] = model_args[argument]
            else:
                continue
        return CoCa(**config)

    else:
        raise NotImplementedError(f"Model type {model_args['model_type']} not implemented")


class UnitTest(ABC):

    def __init__(self):
        self.logger = _setup_logger(logger_name=str(self), log_level=logging.DEBUG)

    def __str__(self) -> str:
        name = self.__class__.__name__
        if name == 'UnitTest':
            name = 'Generic'
        return f"UnitTest(type={name})"

    def __repr__(self) -> str:
        return self.__str__()

    def msg(self, message: str):
        process_name = current_process().name
        if (process_name == 'MainProcess') or (process_name == 'SpawnProcess-1'):
            self.logger.debug(message)

    @staticmethod
    @abstractmethod
    def get_unit_test_dataset(dataset: AF2SCN) -> AF2SCN:
        pass


class Overfit(UnitTest):

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_unit_test_dataset(dataset: AF2SCN, num_samples: int) -> AF2SCN:
        new_manifest = {}
        af2_count, scn_count = 0, 0
        max_per_data_source = num_samples // 2
        for k, v in dataset.manifest.items():
            if len(v['sequence']) <= dataset.max_len:
                if k.split('-')[0] == 'AF':
                    if af2_count < max_per_data_source:
                        new_manifest[k] = v
                        af2_count += 1
                    else:
                        continue
                else:
                    if scn_count < max_per_data_source:
                        new_manifest[k] = v
                        scn_count += 1
                    else:
                        continue
            else:
                continue
        dataset._overwrite_manifest(new_manifest=new_manifest)
        return dataset

    def get_unit_test_model(self, alphabet, devices, model_args):
        self.msg("Unit test model hook")
        self.msg(f"Loading model {model_args['model_type']}")
        return load_model(devices, alphabet, model_args)
