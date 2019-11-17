#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import os
import random
import pickle

import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_sequence, pad_sequence

from utils import parse_scps, stft, apply_cmvn, EPSILON, get_logger

logger = get_logger(__name__)


class SpectrogramReader(object):
    """
        Wrapper for short-time fourier transform of dataset
    """

    def __init__(self, wave_scp, **kwargs):
        if not os.path.exists(wave_scp):
            raise FileNotFoundError("Could not find file {}".format(wave_scp))
        self.stft_kwargs = kwargs
        self.wave_dict = parse_scps(wave_scp)
        self.wave_keys = [key for key in self.wave_dict.keys()]
        logger.info(
            "Create SpectrogramReader for {} with {} utterances".format(
                wave_scp, len(self.wave_dict)))

    def __len__(self):
        return len(self.wave_dict)

    def __contains__(self, key):
        return key in self.wave_dict

    # stft
    def _load(self, key):
        return stft(self.wave_dict[key], **self.stft_kwargs)

    '''
    # sequential index
    def __iter__(self):
        for key in self.wave_dict:
            yield key, self._load(key)
    '''
    # random index

    def __getitem__(self, key):
        if key not in self.wave_dict:
            raise KeyError("Could not find utterance {}".format(key))
        return self._load(key)


class Datasets(object):
    def __init__(self, mix_reader, target_reader_list, mvn_dict='cmvn.dict', apply_log=True):
        self.mix_reader = mix_reader
        self.target_reader_list = target_reader_list
        self.key_list = mix_reader.wave_keys
        self.num_spks = len(target_reader_list)
        self.mvn_dict = mvn_dict
        self.apply_log = apply_log

        if mvn_dict:
            logger.info("Using cmvn dictionary from {}".format(mvn_dict))
            with open(mvn_dict, "rb") as f:
                self.mvn_dict = pickle.load(f)

    def __len__(self):
        return len(self.mix_reader)

    def _has_target(self, key):
        for target in self.target_reader_list:
            if key not in target:
                return False
        return True

    def _transform(self, mixture_specs, targets_specs_list):
        """
        Transform original spectrogram
            If mixture_specs is a complex object, it means PAM will be used for training
            It can be configured in .yaml, egs: apply_abs=false to produce complex results
            If mixture_specs is real, we will using AM(ratio mask)

        Arguments:
            mixture_specs: non-log complex/real spectrogram
            targets_specs_list: list of non-log complex/real spectrogram for each target speakers
        Returns:
            python dictionary with four attributes:
            num_frames: length of current utterance
            feature: input feature for networks, egs: log spectrogram + cmvn
            source_attr: a dictionary with at most 2 keys: spectrogram and phase(for PSM), each contains a tensor
            target_attr: same keys like source_attr, each keys correspond to a tensor list
        """
        # NOTE: mixture_specs may be complex or real
        input_spectra = np.abs(mixture_specs) if np.iscomplexobj(
            mixture_specs) else mixture_specs
        # apply_log and cmvn, for nnet input
        if self.apply_log:
            input_spectra = np.log(np.maximum(input_spectra, EPSILON))
        if self.mvn_dict:
            input_spectra = apply_cmvn(input_spectra, self.mvn_dict)

        # using dict to pack infomation needed in loss
        source_attr = {}
        target_attr = {}

        if np.iscomplexobj(mixture_specs):
            source_attr["spectrogram"] = th.tensor(
                np.abs(mixture_specs), dtype=th.float32)
            target_attr["spectrogram"] = [
                th.tensor(np.abs(t), dtype=th.float32)
                for t in targets_specs_list
            ]
            source_attr["phase"] = th.tensor(
                np.angle(mixture_specs), dtype=th.float32)
            target_attr["phase"] = [
                th.tensor(np.angle(t), dtype=th.float32)
                for t in targets_specs_list
            ]
        else:
            source_attr["spectrogram"] = th.tensor(
                mixture_specs, dtype=th.float32)
            target_attr["spectrogram"] = [
                th.tensor(t, dtype=th.float32) for t in targets_specs_list
            ]

        return {
            "num_frames": mixture_specs.shape[0],
            "feature": th.tensor(input_spectra, dtype=th.float32),
            "source_attr": source_attr,
            "target_attr": target_attr
        }

    def __getitem__(self, index):
        key = self.key_list[index]
        mix = self.mix_reader[key]
        if self._has_target(key):
            ref = [reader[key] for reader in self.target_reader_list]
        else:
            raise ValueError('Not have Target Data')
        
        mix = mix.astype(np.float32)
        ref = [spk.astype(np.float32) for spk in ref]

        return self._transform(mix,ref)
