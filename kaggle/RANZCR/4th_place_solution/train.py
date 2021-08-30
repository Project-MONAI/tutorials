import argparse
import gc
import importlib
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from models.seg_model import RanzcrNet
from utils import (
    batch_to_device,
    create_checkpoint,
    fast_auc,
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

    # set model
    device = "cuda:%d" % cfg.gpu
    cfg.device = device

    model = RanzcrNet(cfg)
    model.to(device)

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
            val_loss, avg_roc = run_eval(
                model=model,
                val_dataloader=val_dataloader,
                cfg=cfg,
                writer=writer,
                epoch=epoch,
            )

        if cfg.train_val is True:
            if (epoch + 1) % cfg.eval_train_epochs == 0 or (epoch + 1) == cfg.epochs:
                train_val_loss, train_avg_roc = run_eval(
                    model, train_val_dataloader, cfg, writer, epoch
                )
                print(
                    f"train_val_loss {train_val_loss:.5} train_avg_roc {train_avg_roc:.5}"
                )

        if val_loss < best_val_loss:
            print(
                f"SAVING CHECKPOINT: val_loss {best_val_loss:.5} -> {val_loss:.5} avg_roc {avg_roc:.5}"
            )
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
        data = next(tr_it)
        iteration += 1

        step += cfg.batch_size
        torch.set_grad_enabled(True)
        batch = batch_to_device(data, cfg.device)

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

            progress_bar.set_description(
                f"loss: {np.mean(losses[-10:]):.2f}"
            )


def run_eval(model, val_dataloader, cfg, writer, epoch):

    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    val_losses = []
    val_preds = []
    val_targets = []

    for data in val_dataloader:

        batch = batch_to_device(data, cfg.device)

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

    rocs = [
        fast_auc(val_targets[:, i], val_preds[:, i]) for i in range(cfg.num_classes)
    ]
    avg_roc = np.mean(rocs)

    writer.add_scalar("val_loss", val_loss, epoch)
    writer.add_scalar("val_avg_roc", avg_roc, epoch)

    return val_loss, avg_roc


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
