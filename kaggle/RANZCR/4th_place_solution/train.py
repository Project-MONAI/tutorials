import argparse
import gc
import importlib
import os
import sys

import numpy as np
import pandas as pd
import torch
from monai.metrics import compute_roc_auc
from monai.transforms import ToDeviced
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.seg_model import RanzcrNet
from utils import (
    create_checkpoint,
    get_optimizer,
    get_scheduler,
    get_train_dataloader,
    get_train_dataset,
    get_val_dataloader,
    get_val_dataset,
    set_seed,
)


def main(cfg):

    os.makedirs(str(cfg.output_dir + f"/fold{cfg.fold}/"), exist_ok=True)

    # set random seed, works when use all data to train
    if cfg.seed < 0:
        cfg.seed = np.random.randint(1_000_000)
    set_seed(cfg.seed)

    # set dataset, dataloader
    train = pd.read_csv(cfg.train_df)

    if cfg.fold == -1:
        val_df = train[train["fold"] == 0]
    else:
        val_df = train[train["fold"] == cfg.fold]
    train_df = train[train["fold"] != cfg.fold]

    train_dataset = get_train_dataset(train_df, cfg)
    val_dataset = get_val_dataset(val_df, cfg)

    train_dataloader = get_train_dataloader(train_dataset, cfg)
    val_dataloader = get_val_dataloader(val_dataset, cfg)

    if cfg.train_val is True:
        train_val_dataset = get_val_dataset(train_df, cfg)
        train_val_dataloader = get_val_dataloader(train_val_dataset, cfg)

    to_device_transform = ToDeviced(
        keys=("input", "target", "mask", "is_annotated"), device=cfg.device
    )
    cfg.to_device_transform = to_device_transform
    # set model

    model = RanzcrNet(cfg)
    model.to(cfg.device)

    # set optimizer, lr scheduler
    total_steps = len(train_dataset)

    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(cfg, optimizer, total_steps)

    # set other tools
    if cfg.mixed_precision:
        scaler = GradScaler()
    else:
        scaler = None

    writer = SummaryWriter(str(cfg.output_dir + f"/fold{cfg.fold}/"))

    # train and val loop
    step = 0
    i = 0
    best_val_loss = np.inf
    optimizer.zero_grad()
    for epoch in range(cfg.epochs):
        print("EPOCH:", epoch)
        gc.collect()
        if cfg.train is True:
            run_train(
                model=model,
                train_dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                scaler=scaler,
                writer=writer,
                epoch=epoch,
                iteration=i,
                step=step,
            )

        if (epoch + 1) % cfg.eval_epochs == 0 or (epoch + 1) == cfg.epochs:
            val_loss = run_eval(
                model=model,
                val_dataloader=val_dataloader,
                cfg=cfg,
                writer=writer,
                epoch=epoch,
            )

        if cfg.train_val is True:
            if (epoch + 1) % cfg.eval_train_epochs == 0 or (epoch + 1) == cfg.epochs:
                train_val_loss = run_eval(
                    model, train_val_dataloader, cfg, writer, epoch
                )
                print(f"train_val_loss {train_val_loss:.5}")

        if val_loss < best_val_loss:
            print(f"SAVING CHECKPOINT: val_loss {best_val_loss:.5} -> {val_loss:.5}")
            best_val_loss = val_loss

            checkpoint = create_checkpoint(
                model,
                optimizer,
                epoch,
                scheduler=scheduler,
                scaler=scaler,
            )
            torch.save(
                checkpoint,
                f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_best_seed{cfg.seed}.pth",
            )


def run_train(
    model,
    train_dataloader,
    optimizer,
    scheduler,
    cfg,
    scaler,
    writer,
    epoch,
    iteration,
    step,
):
    model.train()
    losses = []
    progress_bar = tqdm(range(len(train_dataloader)))
    tr_it = iter(train_dataloader)

    for itr in progress_bar:
        batch = next(tr_it)
        batch = cfg.to_device_transform(batch)
        iteration += 1

        step += cfg.batch_size
        torch.set_grad_enabled(True)

        if cfg.mixed_precision:
            with autocast():
                output_dict = model(batch)
        else:
            output_dict = model(batch)

        loss = output_dict["loss"]
        losses.append(loss.item())

        # Backward pass
        if cfg.mixed_precision:
            scaler.scale(loss).backward()
            if iteration % cfg.grad_accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if iteration % cfg.grad_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        if step % cfg.batch_size == 0:

            progress_bar.set_description(f"loss: {np.mean(losses[-10:]):.2f}")


def run_eval(model, val_dataloader, cfg, writer, epoch):

    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    val_losses = []

    if cfg.compute_auc is True:
        val_preds = []
        val_targets = []

    for batch in val_dataloader:
        batch = cfg.to_device_transform(batch)
        if cfg.mixed_precision:
            with autocast():
                output = model(batch)
        else:
            output = model(batch)

        val_losses += [output["loss"]]
        if cfg.compute_auc is True:
            val_preds += [output["logits"].sigmoid()]
            val_targets += [batch["target"]]

    val_losses = torch.stack(val_losses)
    val_losses = val_losses.cpu().numpy()
    val_loss = np.mean(val_losses)

    if cfg.compute_auc is True:

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_preds = val_preds.cpu().numpy().astype(np.float32)
        val_targets = val_targets.cpu().numpy().astype(np.float32)
        avg_auc = compute_roc_auc(val_preds, val_targets, average="macro")
        writer.add_scalar("val_avg_auc", avg_auc, epoch)

    writer.add_scalar("val_loss", val_loss, epoch)

    return val_loss


if __name__ == "__main__":

    sys.path.append("configs")
    sys.path.append("models")
    sys.path.append("data")

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-c", "--config", help="config filename")
    parser.add_argument("-f", "--fold", type=int, default=-1, help="fold")
    parser.add_argument("-s", "--seed", type=int, default=-1, help="fold")

    parser_args, _ = parser.parse_known_args(sys.argv)

    cfg = importlib.import_module(parser_args.config).cfg

    if parser_args.fold > -1:
        cfg.fold = parser_args.fold
        cfg.seed = parser_args.seed

    main(cfg)
