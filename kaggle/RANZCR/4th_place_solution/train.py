import os

import numpy as np
import importlib
import sys
import random
from tqdm import tqdm
import gc
import argparse

import torch
from torch.utils.data import SequentialSampler, DataLoader
from torch import optim
from torch.cuda.amp import GradScaler, autocast

from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from models.seg_model import Net
from data.seg_data import batch_to_device, CustomDataset
from numba import jit


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_train_dataset(train_df, cfg):
    print("Loading train dataset")

    train_dataset = CustomDataset(train_df, cfg, aug=cfg.train_aug, mode="train")
    return train_dataset


def get_train_dataloader(train_ds, cfg):

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
    print("Loading val dataset")
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


@jit
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def run_eval(model, val_dataloader, cfg, writer, epoch, pre="val"):

    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    val_losses = []
    val_preds = []
    val_targets = []

    for data in tqdm(val_dataloader):

        batch = batch_to_device(data, device)

        if cfg.mixed_precision:
            with autocast():
                output = model(batch)
        else:
            output = model(batch)

        val_losses += [output["loss"]]
        val_preds += [output["logits"].sigmoid()]
        val_targets += [batch["target"]]

    val_losses = torch.stack(val_losses)
    val_preds = torch.cat(val_preds)
    val_targets = torch.cat(val_targets)

    val_losses = val_losses.cpu().numpy()
    val_loss = np.mean(val_losses)

    val_preds = val_preds.cpu().numpy().astype(np.float32)
    val_targets = val_targets.cpu().numpy().astype(np.float32)

    rocs = [fast_auc(val_targets[:, i], val_preds[:, i]) for i in range(cfg.num_classes)]
    avg_roc = np.mean(rocs)

    print(f"{pre}_loss", val_loss)
    print(f"{pre}_avg_roc", avg_roc)

    writer.add_scalar(f"{pre}_loss", val_loss, epoch)
    writer.add_scalar(f"{pre}_avg_roc", avg_roc, epoch)

    print("EVAL FINISHED")

    return val_loss


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


if __name__ == "__main__":

    sys.path.append("configs")
    sys.path.append("models")
    sys.path.append("data")

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-C", "--config", help="config filename")
    parser.add_argument("-f", "--fold", type=int, default=-1, help="fold")
    parser.add_argument("-s", "--seed", type=int, default=-1, help="fold")

    parser_args, _ = parser.parse_known_args(sys.argv)

    cfg = importlib.import_module(parser_args.config).cfg

    if parser_args.fold > -1:
        cfg.fold = parser_args.fold

    if parser_args.fold > -1:
        cfg.seed = parser_args.seed

    os.makedirs(str(cfg.output_dir + f"/fold{cfg.fold}/"), exist_ok=True)

    # set seed
    if cfg.seed < 0:
        cfg.seed = np.random.randint(1_000_000)
    set_seed(cfg.seed)

    writer = SummaryWriter(str(cfg.output_dir + f"/fold{cfg.fold}/"))

    device = "cuda:%d" % cfg.gpu
    cfg.device = device

    # setup dataset
    train = pd.read_csv(cfg.train_df)

    if cfg.fold == -1:
        val_df = train[train["fold"] == 0]
    else:
        val_df = train[train["fold"] == cfg.fold]
    train_df = train[train["fold"] != cfg.fold]

    train_dataset = get_train_dataset(train_df, cfg)
    val_dataset = get_val_dataset(val_df, cfg)
    train_val_dataset = get_val_dataset(train_df, cfg)

    train_dataloader = get_train_dataloader(train_dataset, cfg)
    val_dataloader = get_val_dataloader(val_dataset, cfg)
    train_val_dataloader = get_val_dataloader(train_val_dataset, cfg)

    model = Net(cfg)
    model.to(device)

    total_steps = len(train_dataset)

    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(cfg, optimizer, total_steps)

    if cfg.mixed_precision:
        scaler = GradScaler()
    else:
        scaler = None

    step = 0
    i = 0
    best_val_loss = np.inf
    optimizer.zero_grad()
    for epoch in range(cfg.epochs):

        print("EPOCH:", epoch)

        progress_bar = tqdm(range(len(train_dataloader)))
        tr_it = iter(train_dataloader)

        losses = []

        gc.collect()

        if cfg.train:
            # ==== TRAIN LOOP
            for itr in progress_bar:
                i += 1

                step += cfg.batch_size

                try:
                    data = next(tr_it)
                except Exception as e:
                    print(e)
                    print("DATA FETCH ERROR")
                    # continue

                model.train()
                torch.set_grad_enabled(True)

                # Forward pass

                batch = batch_to_device(data, device)

                if cfg.mixed_precision:
                    with autocast():
                        output_dict = model(batch)
                else:
                    output_dict = model(batch)

                loss = output_dict["loss"]
                cls_loss = output_dict["cls_loss"]
                seg_loss = output_dict["seg_loss"]

                losses.append(loss.item())

                # Backward pass
                if cfg.mixed_precision:
                    scaler.scale(loss).backward()
                    if i % cfg.grad_accumulation == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if i % cfg.grad_accumulation == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                if step % cfg.batch_size == 0:

                    progress_bar.set_description(
                        f"loss: {np.mean(losses[-10:]):.2f}"
                    )

        if (epoch+1) % cfg.eval_epochs == 0 or (epoch+1) == cfg.epochs:
            val_loss = run_eval(model, val_dataloader, cfg, writer, epoch)
        else:
            val_score = 0

        if cfg.train_val is True:
            if (epoch+1) % cfg.eval_train_epochs == 0 or (epoch+1) == cfg.epochs:
                train_val_loss = run_eval(model, train_val_dataloader, cfg, writer, epoch, pre="tr")

        if val_loss < best_val_loss:
            print(f"SAVING CHECKPOINT: val_loss {best_val_loss:.5} -> {val_loss:.5}")
            checkpoint = create_checkpoint(
                model,
                optimizer,
                epoch,
                scheduler=scheduler,
                scaler=scaler,
            )

            torch.save(checkpoint, f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_best_seed{cfg.seed}.pth")
