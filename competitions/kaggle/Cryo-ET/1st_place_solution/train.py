import numpy as np
import pandas as pd
import importlib
import sys
from tqdm import tqdm
import gc
import argparse
import torch
import math
try:
    from torch.amp import GradScaler, autocast
except:
    from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from collections import defaultdict

from utils import (
    sync_across_gpus,
    set_seed,
    get_model,
    create_checkpoint,
    load_checkpoint,
    get_data,
    get_dataset,
    get_dataloader,
    calc_grad_norm,
    calc_weight_norm,
)
from utils import (
    get_optimizer,
    get_scheduler,
    setup_neptune,
    upload_s3,
)


from copy import copy
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import cv2
    cv2.setNumThreads(0)
except:
    print('no cv2 installed, running without')


sys.path.append("configs")
sys.path.append("models")
sys.path.append("data")
sys.path.append("postprocess")
sys.path.append("metrics")


def run_eval(model, val_dataloader, cfg, pre="val", curr_epoch=0):
    saved_images = False
    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    val_data = defaultdict(list)
    val_score =0
    for ind_, data in enumerate(tqdm(val_dataloader, disable=(cfg.local_rank != 0) | cfg.disable_tqdm)):

        batch = cfg.batch_to_device(data, cfg.device)

        if cfg.mixed_precision:
            with autocast('cuda'):
                output = model(batch)
        else:
            output = model(batch)

        if (cfg.local_rank == 0) and (cfg.calc_metric) and (((curr_epoch + 1) % cfg.calc_metric_epochs) == 0):
            # per batch calculations
            pass
        
        if (not saved_images) & (cfg.save_first_batch_preds):
            save_first_batch_preds(batch, output, cfg)
            saved_images = True

        for key, val in output.items():
            val_data[key] += [output[key]]

    for key, val in output.items():
        value = val_data[key]
        if isinstance(value[0], list):
            val_data[key] = [item for sublist in value for item in sublist]
        
        else:
            if len(value[0].shape) == 0:
                val_data[key] = torch.stack(value)
            else:
                val_data[key] = torch.cat(value, dim=0)

    if (cfg.local_rank == 0) and (cfg.calc_metric) and (((curr_epoch + 1) % cfg.calc_metric_epochs) == 0):
        pass

    if cfg.distributed and cfg.eval_ddp:
        for key, val in output.items():
            val_data[key] = sync_across_gpus(val_data[key], cfg.world_size)

    if cfg.local_rank == 0:
        if cfg.save_val_data:
            if cfg.distributed:
                for k, v in val_data.items():
                    val_data[k] = v[: len(val_dataloader.dataset)]
            torch.save(val_data, f"{cfg.output_dir}/fold{cfg.fold}/{pre}_data_seed{cfg.seed}.pth")

    loss_names = [key for key in output if 'loss' in key]
    loss_names += [key for key in output if 'score' in key]
    for k in loss_names:
        if cfg.local_rank == 0 and k in val_data:
            losses = val_data[k].cpu().numpy()
            loss = np.mean(losses)

            print(f"Mean {pre}_{k}", loss)
            if cfg.neptune_run:
                if not math.isinf(loss) and not math.isnan(loss):
                    cfg.neptune_run[f"{pre}/{k}"].log(loss, step=cfg.curr_step)


    if (cfg.local_rank == 0) and (cfg.calc_metric) and (((curr_epoch + 1) % cfg.calc_metric_epochs) == 0):

        val_df = val_dataloader.dataset.df
        pp_out = cfg.post_process_pipeline(cfg, val_data, val_df)
        val_score = cfg.calc_metric(cfg, pp_out, val_df, pre)
        if type(val_score)!=dict:
            val_score = {f'score':val_score}
            
        for k, v in val_score.items():
            print(f"{pre}_{k}: {v:.3f}")
            if cfg.neptune_run:
                if not math.isinf(v) and not math.isnan(v):
                    cfg.neptune_run[f"{pre}/{k}"].log(v, step=cfg.curr_step)
        
    if cfg.distributed:
        torch.distributed.barrier()

#     print("EVAL FINISHED")

    return val_score


def train(cfg):
        # set seed
    if cfg.seed < 0:
        cfg.seed = np.random.randint(1_000_000)
    print("seed", cfg.seed)


    if cfg.distributed:

        cfg.local_rank = int(os.environ["LOCAL_RANK"])
        device = "cuda:%d" % cfg.local_rank
        cfg.device = device

        

        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        cfg.world_size = torch.distributed.get_world_size()
        cfg.rank = torch.distributed.get_rank()
#         print("Training in distributed mode with multiple processes, 1 GPU per process.")
        print(f"Process {cfg.rank}, total {cfg.world_size}, local rank {cfg.local_rank}.")
        cfg.group = torch.distributed.new_group(np.arange(cfg.world_size))
#         print("Group", cfg.group)

        # syncing the random seed
        cfg.seed = int(
            sync_across_gpus(torch.Tensor([cfg.seed]).to(device), cfg.world_size)
            .detach()
            .cpu()
            .numpy()[0]
        )  #
        
        print(f"LOCAL_RANK {cfg.local_rank}, device {device}, seed {cfg.seed}")

    else:
        cfg.local_rank = 0
        cfg.world_size = 1
        cfg.rank = 0  # global rank

        device = "cuda:%d" % cfg.gpu
        cfg.device = device

    set_seed(cfg.seed)

    if cfg.local_rank == 0:
        cfg.neptune_run = setup_neptune(cfg)

    train_df, val_df, test_df = get_data(cfg)
    
    train_dataset = get_dataset(train_df, cfg, mode='train')
    train_dataloader = get_dataloader(train_dataset, cfg, mode='train')
    
    val_dataset = get_dataset(val_df, cfg, mode='val')
    val_dataloader = get_dataloader(val_dataset, cfg, mode='val')
    
    if cfg.test:
        test_dataset = get_dataset(test_df, cfg, mode='test')
        test_dataloader = get_dataloader(test_dataset, cfg, mode='test')

    if cfg.train_val:
        train_val_dataset = get_dataset(train_df, cfg, mode='val')
        train_val_dataloader = get_dataloader(train_val_dataset, cfg, 'val')

    model = get_model(cfg, train_dataset)
    if cfg.compile_model:
        print('compiling model')
        model = torch.compile(model)
    model.to(device)

    if cfg.distributed:

        if cfg.syncbn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = NativeDDP(
            model, device_ids=[cfg.local_rank], find_unused_parameters=cfg.find_unused_parameters
        )

    total_steps = len(train_dataset)
    if train_dataloader.sampler is not None:
        if 'WeightedRandomSampler' in str(train_dataloader.sampler.__class__):
            total_steps = train_dataloader.sampler.num_samples

    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(cfg, optimizer, total_steps)

    if cfg.mixed_precision:
        scaler = GradScaler()
    else:
        scaler = None

    cfg.curr_step = 0
    i = 0
    best_val_loss = np.inf
    optimizer.zero_grad()
    total_grad_norm = None    
    total_weight_norm = None  
    total_grad_norm_after_clip = None

    for epoch in range(cfg.epochs):

        set_seed(cfg.seed + epoch + cfg.local_rank)

        cfg.curr_epoch = epoch
        if cfg.local_rank == 0: 
            print("EPOCH:", epoch)

        
        if cfg.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        progress_bar = tqdm(range(len(train_dataloader)),disable=cfg.disable_tqdm)
        tr_it = iter(train_dataloader)

        losses = []

        gc.collect()

        if cfg.train:
            # ==== TRAIN LOOP
            for itr in progress_bar:
                i += 1

                cfg.curr_step += cfg.batch_size * cfg.world_size

                try:
                    data = next(tr_it)
                except Exception as e:
                    print(e)
                    print("DATA FETCH ERROR")
                    # continue


                model.train()
                torch.set_grad_enabled(True)


                batch = cfg.batch_to_device(data, device)

                if cfg.mixed_precision:
                    with autocast('cuda'):
                        output_dict = model(batch)
                else:
                    if cfg.bf16:
                        with autocast('cuda',dtype=torch.bfloat16):
                            output_dict = model(batch)
                    else:
                        output_dict = model(batch)

                loss = output_dict["loss"]

                losses.append(loss.item())

                if cfg.grad_accumulation >1:
                    loss /= cfg.grad_accumulation

                # Backward pass

                if cfg.mixed_precision:
                    scaler.scale(loss).backward()

                    if i % cfg.grad_accumulation == 0:
                        if (cfg.track_grad_norm) or (cfg.clip_grad > 0):
                            scaler.unscale_(optimizer)
                        if cfg.track_grad_norm:
                            total_grad_norm = calc_grad_norm(model.parameters(), cfg.grad_norm_type)                              
                        if cfg.clip_grad > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                        if cfg.track_grad_norm:
                            total_grad_norm_after_clip = calc_grad_norm(model.parameters(), cfg.grad_norm_type)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        
                else:

                    loss.backward()
                    if i % cfg.grad_accumulation == 0:
                        if cfg.track_grad_norm:
                            total_grad_norm = calc_grad_norm(model.parameters())
                        if cfg.clip_grad > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                        if cfg.track_grad_norm:
                            total_grad_norm_after_clip = calc_grad_norm(model.parameters(), cfg.grad_norm_type)
                        if cfg.track_weight_norm:
                            total_weight_norm = calc_weight_norm(model.parameters(), cfg.grad_norm_type)
                        optimizer.step()
                        optimizer.zero_grad()
                        # print(optimizer.state_dict())
                        # break

                if cfg.distributed:
                    torch.cuda.synchronize()

                if scheduler is not None:
                    scheduler.step()

                if cfg.local_rank == 0 and cfg.curr_step % cfg.batch_size == 0:

                    loss_names = [key for key in output_dict if 'loss' in key]
                    if cfg.neptune_run:
                        for l in loss_names:
                            v = output_dict[l].item()
                            if not math.isinf(v) and not math.isnan(v):
                                cfg.neptune_run[f"train/{l}"].log(value=v, step=cfg.curr_step)
                        cfg.neptune_run["lr"].log(value=optimizer.param_groups[0]["lr"], step=cfg.curr_step)
                        if total_grad_norm is not None:
                            cfg.neptune_run["total_grad_norm"].log(value=total_grad_norm.item(), step=cfg.curr_step)
                            cfg.neptune_run["total_grad_norm_after_clip"].log(value=total_grad_norm_after_clip.item(), step=cfg.curr_step)
                        if total_weight_norm is not None:
                            cfg.neptune_run["total_weight_norm"].log(value=total_weight_norm.item(), step=cfg.curr_step)
                    progress_bar.set_description(f"loss: {np.mean(losses[-10:]):.4f}")

                if cfg.eval_steps != 0:
                    if i % cfg.eval_steps == 0:
                        if cfg.distributed and cfg.eval_ddp:
                            val_loss = run_eval(model, val_dataloader, cfg, pre="val", curr_epoch=epoch)
                        else:
                            if cfg.local_rank == 0:
                                val_loss = run_eval(model, val_dataloader, cfg, pre="val", curr_epoch=epoch)
                    else:
                        val_score = 0

            print(f"Mean train_loss {np.mean(losses):.4f}")

        if cfg.distributed:
            torch.cuda.synchronize()
        if cfg.force_fp16:
            model = model.half().float()
        if cfg.val:

            if (epoch + 1) % cfg.eval_epochs == 0 or (epoch + 1) == cfg.epochs:
                if cfg.distributed and cfg.eval_ddp:
                    val_score = run_eval(model, val_dataloader, cfg, pre="val", curr_epoch=epoch)
                else:
                    if cfg.local_rank == 0:
                        val_score = run_eval(model, val_dataloader, cfg, pre="val", curr_epoch=epoch)
            else:
                val_score = 0
            
        if cfg.train_val == True:
            if (epoch + 1) % cfg.eval_train_epochs == 0 or (epoch + 1) == cfg.epochs:
                if cfg.distributed and cfg.eval_ddp:
                    _ = get_preds(model, train_val_dataloader, cfg, pre=cfg.pre_train_val)

                else:
                    if cfg.local_rank == 0:
                        _ = get_preds(model, train_val_dataloader, cfg, pre=cfg.pre_train_val)


        if cfg.distributed:
            torch.distributed.barrier()

        if (cfg.local_rank == 0) and (cfg.epochs > 0) and (cfg.save_checkpoint):
            if not cfg.save_only_last_ckpt:
                checkpoint = create_checkpoint(cfg, model, optimizer, epoch, scheduler=scheduler, scaler=scaler)

                torch.save(
                    checkpoint, f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pth"
                )

    if (cfg.local_rank == 0) and (cfg.epochs > 0) and (cfg.save_checkpoint):
        checkpoint = create_checkpoint(cfg, model, optimizer, epoch, scheduler=scheduler, scaler=scaler)

        torch.save(
            checkpoint, f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pth"
        )

    if cfg.test:
        run_eval(model, test_dataloader, test_df, cfg, pre="test")

    return val_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument("-C", "--config", help="config filename")
    parser.add_argument("-D", "--debug", action='store_true', help="debugging True/ False")
    parser_args, other_args = parser.parse_known_args(sys.argv)

    cfg = copy(importlib.import_module(parser_args.config).cfg)

    if parser_args.debug:
        print('debug mode')
        cfg.neptune_connection_mode = 'debug'
        
        
    # overwrite params in config with additional args
    if len(other_args) > 1:
        other_args = {k.replace('-',''):v for k, v in zip(other_args[1::2], other_args[2::2])}

        for key in other_args:
            if key in cfg.__dict__:

                print(f'overwriting cfg.{key}: {cfg.__dict__[key]} -> {other_args[key]}')
                cfg_type = type(cfg.__dict__[key])
                if other_args[key] == 'None':
                    cfg.__dict__[key] = None
                elif cfg_type == bool:
                    cfg.__dict__[key] = other_args[key] == 'True'
                elif cfg_type == type(None):
                    cfg.__dict__[key] = other_args[key]
                else:
                    cfg.__dict__[key] = cfg_type(other_args[key])


    os.makedirs(str(cfg.output_dir + f"/fold{cfg.fold}/"), exist_ok=True)

    cfg.CustomDataset = importlib.import_module(cfg.dataset).CustomDataset
    cfg.tr_collate_fn = importlib.import_module(cfg.dataset).tr_collate_fn
    cfg.val_collate_fn = importlib.import_module(cfg.dataset).val_collate_fn
    cfg.batch_to_device = importlib.import_module(cfg.dataset).batch_to_device

    cfg.post_process_pipeline = importlib.import_module(cfg.post_process_pipeline).post_process_pipeline
    cfg.calc_metric = importlib.import_module(cfg.metric).calc_metric
    
    result = train(cfg)
    print(result)
