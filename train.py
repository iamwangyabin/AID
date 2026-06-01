import os
import hydra
import argparse
import datetime

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

import engine
import data
import networks
from utils.common import load_config_with_cli, archive_files, seed_everything
from utils.looger import build_experiment_logger


def resolve_target(path):
    if callable(path):
        return path
    if not isinstance(path, str):
        raise TypeError(f"Config target must be a string or callable, got {type(path).__name__}.")
    if hasattr(hydra.utils, "get_object"):
        target = hydra.utils.get_object(path)
    else:
        try:
            target = hydra.utils.get_class(path)
        except Exception:
            target = hydra.utils.get_method(path)
    if not callable(target):
        raise TypeError(f'Config target "{path}" resolved to non-callable {type(target).__name__}.')
    return target


def build_dataloader(conf):
    train_datasets = []
    for sub_data in conf.datasets.train.source:
        dataset_cls = resolve_target(sub_data.target)
        for sub_set in sub_data.sub_sets:
            train_data = dataset_cls(
                sub_data.data_root,
                conf.datasets.train.trsf,
                subset=sub_set,
                split=sub_data.split,
            )
            train_datasets.append(train_data)
    train_datasets = ConcatDataset(train_datasets)

    val_datasets = []
    for sub_data in conf.datasets.val.source:
        dataset_cls = resolve_target(sub_data.target)
        for sub_set in sub_data.sub_sets:
            val_data = dataset_cls(
                sub_data.data_root,
                conf.datasets.val.trsf,
                subset=sub_set,
                split=sub_data.split,
            )
            val_datasets.append(val_data)
    val_datasets = ConcatDataset(val_datasets)

    train_loader = DataLoader(train_datasets, batch_size=conf.datasets.train.batch_size, shuffle=True,
                              num_workers=conf.datasets.train.loader_workers)
    val_loader = DataLoader(val_datasets, batch_size=conf.datasets.val.batch_size, shuffle=False,
                            num_workers=conf.datasets.val.loader_workers)
    return train_loader, val_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--cfg', type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()
    conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    conf = hydra.utils.instantiate(conf)

    seed_everything(conf.train.seed)

    train_loader, val_loader = build_dataloader(conf)
    today_str = conf.name + "_" + datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
    experiment_logger = build_experiment_logger(conf, today_str)

    if os.getenv("LOCAL_RANK", '0') == '0':
        archive_files(today_str, exclude_dirs=['logs', 'wandb', '.git', 'exp_results'])

    model = resolve_target(conf.train.pipeline)(opt=conf)
    model.fit(train_loader, val_loader, experiment_logger=experiment_logger, run_name=today_str)
