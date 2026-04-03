import os
import hydra
import argparse
import wandb
import datetime

import torch
import torch.nn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import engine
import data
import networks
from utils.common import load_config_with_cli, archive_files, seed_everything



def build_dataloader(conf):
    train_datasets = []
    for sub_data in conf.datasets.train.source:
        for sub_set in sub_data.sub_sets:
            train_data = eval(sub_data.target)(sub_data.data_root, conf.datasets.train.trsf,
                                               subset=sub_set, split=sub_data.split)
            train_datasets.append(train_data)
    train_datasets = ConcatDataset(train_datasets)

    val_datasets = []
    for sub_data in conf.datasets.val.source:
        for sub_set in sub_data.sub_sets:
            val_data = eval(sub_data.target)(sub_data.data_root, conf.datasets.val.trsf,
                                          subset=sub_set, split=sub_data.split)
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
    today_str = conf.name +"_"+ datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
    wandb_logger = WandbLogger(name=today_str, project='DeepfakeDetection',
                               job_type='train', group=conf.name)

    # Conditionally sync with swanlab if available
    try:
        import swanlab
        os.environ["WANDB_MODE"]="offline"
        swanlab.sync_wandb()
    except ImportError:
        pass

    if os.getenv("LOCAL_RANK", '0') == '0':
        archive_files(today_str, exclude_dirs=['logs', 'wandb', '.git', 'exp_results'])

    checkpoint_callback = ModelCheckpoint(
        monitor='val_ap_epoch',
        dirpath=os.path.join('logs', today_str),
        filename='{epoch:02d}-{val_ap_epoch:.2f}',
        save_top_k=1,
        mode='max',
    )

    model = eval(conf.train.pipeline)(opt=conf)
    torch.set_float32_matmul_precision('high')
    trainer = L.Trainer(logger=wandb_logger, max_epochs=conf.train.train_epochs, accelerator="gpu", devices=conf.train.gpu_ids,
                        callbacks=[checkpoint_callback],
                        check_val_every_n_epoch=conf.train.check_val_every_n_epoch,
                        num_sanity_val_steps=conf.train.get('num_sanity_val_steps', 2),
                        precision=conf.train.get('precision', "16"))

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.save_checkpoint(os.path.join('logs', today_str, "last.ckpt"))

