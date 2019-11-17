#!/usr/bin/env python
# coding=utf-8

# wujian@2018

import argparse
import os
import torch as th
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from trainer import PITrainer
from dataset import SpectrogramReader, Datasets
from model import PITNet
from utils import nfft, parse_yaml, get_logger
from torch.utils.data import DataLoader
logger = get_logger(__name__)


def _collate(egs):
    """
        Transform utterance index into a minbatch

        Arguments:
            index: a list type [{},{},{}]

        Returns:
            input_sizes: a tensor correspond to utterance length
            input_feats: packed sequence to feed networks
            source_attr/target_attr: dictionary contains spectrogram/phase needed in loss computation
        """
    num_spks = 2 #you need to set this paramater by yourself

    if type(egs) is not list:
        raise ValueError("Unsupported index type({})".format(type(egs)))

    def prepare_target(dict_lsit, index, key):
        return pad_sequence([d["target_attr"][key][index] for d in dict_lsit], batch_first=True)

    dict_list = sorted([eg for eg in egs],
                        key=lambda x: x['num_frames'], reverse=True)

    #input_feats = pack_sequence([d['feature'] for d in dict_list])
    input_feats = pad_sequence([d['feature'] for d in dict_list], batch_first=True)

    input_sizes = th.tensor([d['num_frames']
                                for d in dict_list], dtype=th.float32)

    source_attr = {}
    target_attr = {}

    source_attr['spectrogram'] = pad_sequence(
            [d['source_attr']["spectrogram"] for d in dict_list], batch_first=True)
    target_attr['spectrogram'] = [prepare_target(
            dict_list, index, 'spectrogram') for index in range(num_spks)]

    if 'phase' in dict_list[0]['source_attr'] and 'phase' in dict_list[0]['target_attr']:
        source_attr['phase'] = pad_sequence(
                [d['source_attr']['phase'] for d in dict_list], batch_first=True)
        target_attr['phase'] = [prepare_target(
                dict_list, index, 'phase') for index in range(num_spks)]

    return input_sizes, input_feats, source_attr, target_attr


def uttloader(scp_config, reader_kwargs, loader_kwargs, train=True):
    mix_reader = SpectrogramReader(scp_config['mixture'], **reader_kwargs)
    target_reader = [
        SpectrogramReader(scp_config[spk_key], **reader_kwargs)
        for spk_key in scp_config if spk_key[:3] == 'spk'
    ]
    dataset = Datasets(mix_reader, target_reader)
    # modify shuffle status
    loader_kwargs["shuffle"] = train
    # validate perutt if needed
    # if not train:
    #     loader_kwargs["batch_size"] = 1
    # if validate, do not shuffle
    #utt_loader = DataLoaders(dataset, **loader_kwargs)

    utt_loader = DataLoader(dataset, batch_size=40,shuffle=loader_kwargs['shuffle'],
                            num_workers=10, sampler=None,drop_last=True,
                            collate_fn=_collate)
    return utt_loader


def train(args):
    gpuid = tuple(map(int, args.gpus.split(',')))
    debug = args.debug
    logger.info(
        "Start training in {} model".format('debug' if debug else 'normal'))
    num_bins, config_dict = parse_yaml(args.config)
    reader_conf = config_dict["spectrogram_reader"]
    loader_conf = config_dict["dataloader"]
    dcnnet_conf = config_dict["model"]

    logger.info("Training with {}".format("IRM" if reader_conf["apply_abs"]
                                          else "PSM"))
    batch_size = loader_conf["batch_size"]
    logger.info(
        "Training in {}".format("per utterance" if batch_size == 1 else
                                '{} utterance per batch'.format(batch_size)))

    train_loader = uttloader(
        config_dict["train_scp_conf"]
        if not debug else config_dict["debug_scp_conf"],
        reader_conf,
        loader_conf,
        train=True)
    valid_loader = uttloader(
        config_dict["valid_scp_conf"]
        if not debug else config_dict["debug_scp_conf"],
        reader_conf,
        loader_conf,
        train=False)
    checkpoint = config_dict["trainer"]["checkpoint"]
    logger.info("Training for {} epoches -> {}...".format(
        args.num_epoches, "default checkpoint"
        if checkpoint is None else checkpoint))

    nnet = PITNet(num_bins, **dcnnet_conf)
    trainer = PITrainer(nnet, **config_dict["trainer"], gpuid=gpuid)
    trainer.run(train_loader, valid_loader, num_epoches=args.num_epoches)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Command to start PIT training, configured by .yaml files")
    parser.add_argument(
        "--flags",
        type=str,
        default="",
        help="This option is used to show what this command is runing for")
    parser.add_argument(
        "--config",
        type=str,
        default="train.yaml",
        dest="config",
        help="Location of .yaml configure files for training")
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        dest="debug",
        help="If true, start training in debug data")
    parser.add_argument(
        "--num-epoches",
        type=int,
        default=20,
        dest="num_epoches",
        help="Number of epoches to train")
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="Training on which GPUs "
        "(one or more, egs: 0, \"0,1\")")
    args = parser.parse_args()
    train(args)
