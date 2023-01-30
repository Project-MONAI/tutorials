# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from monai.utils import set_determinism
from torch import optim
from torch.utils.data import DataLoader, SequentialSampler

from data.seg_data import CustomDataset


def set_seed(seed):
    # use monai's function to set the seed.
    # since the function will also change the deterministic settings, which are unnecessary here,
    # we need to modify the values back.
    set_determinism(seed=seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_train_dataset(train_df, cfg):
    train_dataset = CustomDataset(train_df, cfg, aug=cfg.train_aug, mode="train")
    return train_dataset


def get_train_dataloader(train_dataset, cfg):

    train_dataloader = DataLoader(
        train_dataset,
        sampler=None,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=None,
        drop_last=cfg.drop_last,
    )
    print(f"train: dataset {len(train_dataset)}, dataloader {len(train_dataloader)}")
    return train_dataloader


def get_val_dataset(val_df, cfg):
    val_dataset = CustomDataset(val_df, cfg, aug=cfg.val_aug, mode="val")
    return val_dataset


def get_val_dataloader(val_dataset, cfg):

    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=None,
    )
    print(f"valid: dataset {len(val_dataset)}, dataloader {len(val_dataloader)}")
    return val_dataloader


def get_test_dataset(test_df, cfg):
    test_dataset = CustomDataset(test_df, cfg, aug=cfg.test_aug, mode="test")
    return test_dataset


def get_test_dataloader(test_dataset, cfg):

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    return test_dataloader


def get_optimizer(model, cfg):

    params = model.parameters()
    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    return optimizer


def get_scheduler(cfg, optimizer, total_steps):

    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=cfg.warmup * (total_steps // cfg.batch_size),
        t_total=cfg.epochs * (total_steps // cfg.batch_size),
    )

    return scheduler


def create_checkpoint(model, optimizer, epoch, scheduler=None, scaler=None):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint
