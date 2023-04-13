from typing import Optional, Union, Tuple, Dict
from multiprocessing import current_process
from abc import ABC, abstractmethod
from inspect import getfullargspec
import logging
import sys

from torch import nn

from ..utils.data import  AF2SCN
from ..utils.tokenizer import Alphabet
from .experimental import standard_structure_module, CoCa


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
    def get_unit_test_dataset(dataset: AF2SCN) -> AF2SCN:
        new_manifest = {}
        af2_count, scn_count = 0, 0
        for k, v in dataset.manifest.items():
            if len(v['sequence']) <= 125:
                if k.split('-')[0] == 'AF':
                    if af2_count < 32:
                        new_manifest[k] = v
                        af2_count += 1
                    else:
                        continue
                else:
                    if scn_count < 32:
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
