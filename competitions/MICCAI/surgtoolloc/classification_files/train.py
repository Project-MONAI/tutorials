import argparse
import gc
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from monai.bundle import ConfigParser
from monai.metrics import ConfusionMatrixMetric
from monai.networks.nets import EfficientNetBN
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import (
    SurgDataset,
    create_checkpoint,
    get_train_dataloader,
    get_val_dataloader,
    mixup_data,
    set_seed,
)


def main(cfg):

    os.makedirs(str(cfg.output_dir + f"/fold{cfg.fold}/"), exist_ok=True)
    set_seed(cfg.seed)
    # set dataset, dataloader
    df = pd.read_csv(cfg.train_df)
    sucirr_videos = df[df["suction irrigator"] > 0].clip_name.unique()
    tipup_videos = df[df["tip-up fenestrated grasper"] > 0].clip_name.unique()
    df_oversample = df[
        df.clip_name.isin(sucirr_videos) | df.clip_name.isin(tipup_videos)
    ]

    cfg.labels = df.columns.values[4:18]

    val_df = df[df["fold"] == cfg.fold]
    train_df = pd.concat(
        [df[df["fold"] != cfg.fold]]
        + [df_oversample[df_oversample["fold"] != cfg.fold]] * cfg.oversample_rate
    )

    train_dataset = SurgDataset(cfg, df=train_df, mode="train")
    val_dataset = SurgDataset(cfg, df=val_df, mode="val")

    train_dataloader = get_train_dataloader(train_dataset, cfg)
    val_dataloader = get_val_dataloader(val_dataset, cfg)

    # set model
    model = EfficientNetBN(
        model_name=cfg.backbone, pretrained=True, num_classes=cfg.num_classes
    )
    model = torch.nn.DataParallel(model)
    model.to(cfg.device)

    if cfg.weights is not None:
        model.load_state_dict(
            torch.load(os.path.join(f"{cfg.output_dir}/fold{cfg.fold}", cfg.weights))[
                "model"
            ]
        )
        print(f"weights from: {cfg.weights} are loaded.")

    # set optimizer, lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        epochs=cfg.epochs,
        steps_per_epoch=int(train_df.shape[0] / cfg.batch_size),
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=10**5,
    )
    # set loss, metric
    class_num = list(train_df[cfg.labels].sum())
    class_weights = [
        train_df.shape[0] / (n * cfg.num_classes) if n > 0 else 1 for n in class_num
    ]
    loss_function = torch.nn.BCEWithLogitsLoss(
        weight=torch.as_tensor(class_weights).to(cfg.device)
    )
    metric = ConfusionMatrixMetric(metric_name="F1", reduction="mean_batch")

    # set other tools
    scaler = GradScaler()
    writer = SummaryWriter(str(cfg.output_dir + f"/fold{cfg.fold}/"))

    # train and val loop
    step = 0
    i = 0
    best_metric = run_eval(
        model=model,
        val_dataloader=val_dataloader,
        cfg=cfg,
        writer=writer,
        epoch=-1,
        metric=metric,
    )
    optimizer.zero_grad()
    print("start from: ", best_metric)
    for epoch in range(cfg.epochs):
        print("EPOCH:", epoch)
        gc.collect()
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
            loss_function=loss_function,
        )

        val_metric = run_eval(
            model=model,
            val_dataloader=val_dataloader,
            cfg=cfg,
            writer=writer,
            epoch=epoch,
            metric=metric,
        )

        if val_metric > best_metric:
            print(f"SAVING CHECKPOINT: val_metric {best_metric:.5} -> {val_metric:.5}")
            best_metric = val_metric

            checkpoint = create_checkpoint(
                model,
                optimizer,
                epoch,
                scheduler=scheduler,
                scaler=scaler,
            )
            torch.save(
                checkpoint,
                f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_best_metric.pth",
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
    loss_function,
):
    model.train()
    losses = []
    progress_bar = tqdm(range(len(train_dataloader)))
    tr_it = iter(train_dataloader)

    for itr in progress_bar:
        batch = next(tr_it)
        inputs, labels = batch["input"].to(cfg.device), batch["label"].to(cfg.device)
        iteration += 1

        step += cfg.batch_size
        torch.set_grad_enabled(True)
        if torch.rand(1) > 0.5:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels)
            with autocast():
                outputs = model(inputs)
                loss = lam * loss_function(outputs, labels_a) + (
                    1 - lam
                ) * loss_function(outputs, labels_b)
        else:
            with autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
        losses.append(loss.item())

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        progress_bar.set_description(f"loss: {np.mean(losses):.2f}")


def run_eval(model, val_dataloader, cfg, writer, epoch, metric):

    model.eval()
    torch.set_grad_enabled(False)

    progress_bar = tqdm(range(len(val_dataloader)))
    tr_it = iter(val_dataloader)

    for itr in progress_bar:
        batch = next(tr_it)
        inputs, labels = batch["input"].to(cfg.device), batch["label"].to(cfg.device)
        outputs = model(inputs)
        outputs = (torch.sigmoid(outputs) > cfg.clf_threshold).float()
        metric(outputs, labels)
    score = metric.aggregate()[0]
    print(score)
    score = torch.mean(score).item()
    metric.reset()
    writer.add_scalar("F1", score, epoch)

    return score


if __name__ == "__main__":

    sys.path.append("configs")
    sys.path.append("models")
    sys.path.append("data")

    arg_parser = argparse.ArgumentParser(description="")

    arg_parser.add_argument(
        "-c", "--config", type=str, default="cfg_efnb4.yaml", help="config filename"
    )
    arg_parser.add_argument("-f", "--fold", type=int, default=0, help="fold")
    arg_parser.add_argument("-s", "--seed", type=int, default=-1, help="seed")
    arg_parser.add_argument("-w", "--weights", default=None, help="the path of weights")

    input_args, _ = arg_parser.parse_known_args(sys.argv)

    config_parser = ConfigParser()
    config_parser.read_config(input_args.config)
    config_parser.parse()
    cfg = SimpleNamespace(**config_parser.get_parsed_content("cfg"))

    cfg.fold = input_args.fold
    cfg.seed = input_args.seed
    cfg.weights = input_args.weights

    main(cfg)
