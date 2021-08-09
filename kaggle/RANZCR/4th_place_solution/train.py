import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import numpy as np
import importlib
import sys
import random
from tqdm import tqdm
import gc
import argparse
import math

import torch
from torch.utils.data import Sampler, RandomSampler, SequentialSampler, DataLoader
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import cv2
from monai.optimizers.lr_scheduler import WarmupCosineSchedule

cv2.setNumThreads(0)


sys.path.append('configs')
sys.path.append('models')
sys.path.append('data')
sys.path.append('losses')
sys.path.append('utils')


parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config", help="config filename")
parser.add_argument("-f", "--fold",type=int , default=-1, help="fold")
parser.add_argument("-s", "--seed",type=int , default=-1, help="fold")
parser.add_argument("-debug", "--debug",type=bool , default=False, help="debug mode")
parser_args, _ = parser.parse_known_args(sys.argv)


cfg = importlib.import_module(parser_args.config).cfg

cfg.debug = parser_args.debug

if parser_args.fold > -1:
    cfg.fold = parser_args.fold

if parser_args.fold > -1:
    cfg.seed = parser_args.seed

os.makedirs(str(cfg.output_dir + f'/fold{cfg.fold}/'), exist_ok=True)

CustomDataset = importlib.import_module(cfg.dataset).CustomDataset
tr_collate_fn = importlib.import_module(cfg.dataset).tr_collate_fn
val_collate_fn = importlib.import_module(cfg.dataset).val_collate_fn
batch_to_device = importlib.import_module(cfg.dataset).batch_to_device

class OrderedDistributedSampler(Sampler):

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        print("TOTAL SIZE", self.total_size)

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank*self.num_samples:self.rank*self.num_samples+self.num_samples]
        print("SAMPLES", self.rank*self.num_samples, self.rank*self.num_samples+self.num_samples)
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

def sync_across_gpus(t, world_size):
    torch.distributed.barrier(group)
    gather_t_tensor = [torch.ones_like(t) for _ in range(world_size)]
    torch.distributed.all_gather(gather_t_tensor, t)
    return torch.cat(gather_t_tensor)

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

    if cfg.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=cfg.world_size, rank=cfg.local_rank, shuffle=True, seed=cfg.seed)
    else:
        sampler = None

    train_dataloader = DataLoader(train_dataset,
                                  sampler=sampler,
                                  shuffle=(sampler is None),
                                  batch_size=cfg.batch_size,
                                  num_workers=cfg.num_workers,
                                  pin_memory=False,
                                  collate_fn= tr_collate_fn,
                                  drop_last = cfg.drop_last
                                 )
    print(f"train: dataset {len(train_dataset)}, dataloader {len(train_dataloader)}")
    return train_dataloader

def get_val_dataset(val_df, cfg):
    print("Loading val dataset")
    val_dataset = CustomDataset(val_df, cfg, aug=cfg.val_aug, mode='val')
    return val_dataset

def get_val_dataloader(val_dataset, cfg):

    if cfg.distributed and cfg.eval_ddp:
        sampler = OrderedDistributedSampler(val_dataset, num_replicas=cfg.world_size, rank=cfg.local_rank)
    else:
        sampler = SequentialSampler(val_dataset)

    val_dataloader = DataLoader(val_dataset,
#                                   shuffle=False,
                                  sampler=sampler,
                                  batch_size=cfg.batch_size,
                                  num_workers=cfg.num_workers,
                                  pin_memory=False,
                                collate_fn= val_collate_fn
                                
                                 )
    print(f"valid: dataset {len(val_dataset)}, dataloader {len(val_dataloader)}")
    return val_dataloader



def get_model(cfg):
    Net = importlib.import_module(cfg.model).Net
    return Net(cfg)

def get_optimizer(model, cfg):

    params = model.parameters()
    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    return optimizer


def get_scheduler(cfg, optimizer, total_steps):
        
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=cfg.warmup * (total_steps // cfg.batch_size) // cfg.world_size,
        t_total=cfg.epochs * (total_steps // cfg.batch_size) // cfg.world_size,
    )

    return scheduler

from numba import jit

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
    val_seg_losses = []
    val_seg_losses2 = []
    val_cls_losses = []
    for data in tqdm(val_dataloader, disable=cfg.local_rank != 0):

        batch = batch_to_device(data, device)

        if cfg.mixed_precision:
            with autocast():
                output = model(batch)
        else:
            output = model(batch)

        val_losses += [output['loss']]
        val_preds += [output['logits'].sigmoid()]
        val_targets += [batch['target']]
        
        if 'seg_loss' in output.keys():
            val_cls_losses += [output['cls_loss']]
            val_seg_losses += [output['seg_loss']]
            
        if 'seg_loss2' in output.keys():
            val_seg_losses2 += [output['seg_loss2']]            
        
        
    val_losses = torch.stack(val_losses)
    val_preds = torch.cat(val_preds)
    val_targets = torch.cat(val_targets)
    
    if len(val_seg_losses) > 0:
        val_seg_losses = torch.stack(val_seg_losses)
        val_cls_losses = torch.stack(val_cls_losses)

    if len(val_seg_losses2) > 0:
        val_seg_losses2 = torch.stack(val_seg_losses2)
    
    if cfg.distributed and cfg.eval_ddp:
        val_losses = sync_across_gpus(val_losses, cfg.world_size)
        val_preds = sync_across_gpus(val_preds, cfg.world_size)
        val_targets = sync_across_gpus(val_targets, cfg.world_size)
        
        if len(val_seg_losses) > 0:
            val_seg_losses = sync_across_gpus(val_seg_losses, cfg.world_size)
            val_cls_losses = sync_across_gpus(val_cls_losses, cfg.world_size)

        if len(val_seg_losses2) > 0:
            val_seg_losses2 = sync_across_gpus(val_seg_losses2, cfg.world_size)

    if cfg.local_rank == 0:
        val_losses = val_losses.cpu().numpy()
        val_loss = np.mean(val_losses)
        
        val_preds = val_preds.cpu().numpy().astype(np.float32)
        val_targets = val_targets.cpu().numpy().astype(np.float32)
        if cfg.debug:
            avg_roc = 0.0
        else:
            rocs = [fast_auc(val_targets[:,i], val_preds[:,i]) for i in range(len(val_dataloader.dataset.label_cols))]

            avg_roc = np.mean(rocs)
        
        print(f"{pre}_loss", val_loss)
        print(f"{pre}_avg_roc", avg_roc)

        writer.add_scalar(f"{pre}_loss", val_loss, epoch)
        writer.add_scalar(f"{pre}_avg_roc", avg_roc, epoch)

        if len(val_seg_losses) > 0:
            val_seg_loss = val_seg_losses.cpu().numpy().mean()
            val_cls_loss = val_cls_losses.cpu().numpy().mean() 

            
        if len(val_seg_losses2) > 0:
            val_seg_loss2 = val_seg_losses2.cpu().numpy().mean()

    else:
        val_loss = 0.

    if cfg.distributed:
        torch.distributed.barrier(group)

    print("EVAL FINISHED")

    return val_loss


def create_checkpoint(model, optimizer, epoch, scheduler =None, scaler=None):
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
    
    #set seed
    if cfg.seed < 0:
        cfg.seed = np.random.randint(1_000_000)
    set_seed(cfg.seed)  

    writer = SummaryWriter(str(cfg.output_dir + f'/fold{cfg.fold}/'))
    
    cfg.distributed = False
    if "WORLD_SIZE" in os.environ:
        cfg.distributed = int(os.environ["WORLD_SIZE"]) > 1

    if cfg.distributed:
        
        #NOT SUPPORTED YET
        cfg.local_rank = int(os.environ["LOCAL_RANK"])

        print("RANK",cfg.local_rank)

        device = "cuda:%d" % cfg.local_rank
        cfg.device = device
        print("device", device)
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        cfg.world_size = torch.distributed.get_world_size()
        cfg.rank = torch.distributed.get_rank()
        print("Training in distributed mode with multiple processes, 1 GPU per process.")
        print(f"Process {cfg.rank}, total {cfg.world_size}, local rank {cfg.local_rank}.")
        group = torch.distributed.new_group(np.arange(cfg.world_size))
        print("Group", group)

    else:
        cfg.local_rank = 0
        cfg.world_size = 1
        rank = 0  # global rank

        device = "cuda:%d" % cfg.gpu
        cfg.device = device


    #setup dataset
    train = pd.read_csv(cfg.train_df)
        
    if cfg.do_test:
        test_df = pd.read_csv(cfg.data_dir + 'sample_submission.csv')

    if cfg.fold == -1:
        val_df = train[train['fold'] == 0]
    else:
        val_df = train[train['fold'] == cfg.fold]
    train_df = train[train['fold'] != cfg.fold]

    if cfg.debug:
        train_df = train_df.head(100)
        val_df = val_df.head(20)

    train_dataset = get_train_dataset(train_df,cfg)
    val_dataset = get_val_dataset(val_df,cfg)
    train_val_dataset = get_val_dataset(train_df, cfg)

    train_dataloader = get_train_dataloader(train_dataset, cfg)
    val_dataloader = get_val_dataloader(val_dataset, cfg)
    train_val_dataloader = get_val_dataloader(train_val_dataset, cfg)

    model = get_model(cfg)
    model.to(device)

    if cfg.distributed:

        if cfg.syncbn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = NativeDDP(model, device_ids=[cfg.local_rank], find_unused_parameters=cfg.find_unused_parameters)

    # if cfg.pretrained_weights is not None:
    #     model.load_state_dict(torch.load(cfg.pretrained_weights, map_location='cpu')['model'], strict=True)
    #     print('weights loaded from',cfg.pretrained_weights)

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

        if cfg.epoch_weights is not None:
            model.w = cfg.epoch_weights[epoch]
            print("weight", model.w)

        if cfg.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        progress_bar = tqdm(range(len(train_dataloader)))
        tr_it = iter(train_dataloader)

        losses = []

        gc.collect()

        if cfg.train:
            # ==== TRAIN LOOP
            for itr in progress_bar:
                i += 1

                step += cfg.batch_size * cfg.world_size


                try:
                    data = next(tr_it)
                except Exception as e:
                    print(e)
                    print("DATA FETCH ERROR")
                    # continue

                model.train()
                torch.set_grad_enabled(True)

                

                # Forward pass

                batch = batch_to_device(data,device)

                if cfg.mixed_precision:
                    with autocast():
                        output_dict = model(batch)
                else:
                    output_dict = model(batch)

                loss = output_dict['loss']
                cls_loss = output_dict['cls_loss']
                seg_loss = output_dict['seg_loss']


                losses.append(loss.item())

                # Backward pass
                
                if cfg.mixed_precision:
                    scaler.scale(loss).backward()
                    if cfg.clip_grad > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                    if i % cfg.grad_accumulation == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if cfg.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                    if i % cfg.grad_accumulation == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        
                if cfg.distributed:
                    torch.cuda.synchronize()

                if scheduler is not None:
                    scheduler.step()

                if cfg.local_rank == 0 and step % cfg.batch_size == 0:

                    progress_bar.set_description(
                        f"loss: {np.mean(losses[-10:]):.2f}"
                    )

        #EPOCH END
        #sync at end of epoch

        if cfg.distributed:
            torch.cuda.synchronize() 

        if (epoch+1) % cfg.eval_epochs == 0 or (epoch+1) == cfg.epochs:
            if cfg.distributed and cfg.eval_ddp:
                #torch.cuda.synchronize()
                val_loss = run_eval(model, val_dataloader, cfg, writer, epoch)
            else:
                if cfg.local_rank == 0:
                    val_loss = run_eval(model, val_dataloader, cfg, writer, epoch)
        else:
            val_score = 0

        if cfg.train_val == True:
            if (epoch+1) % cfg.eval_train_epochs == 0 or (epoch+1) == cfg.epochs:
                if cfg.distributed and cfg.eval_ddp:
                    train_val_loss = run_eval(model, train_val_dataloader, cfg, writer, epoch, pre="tr")
                else:
                    if cfg.local_rank == 0:
                        train_val_loss = run_eval(model, train_val_dataloader, cfg, writer, epoch, pre="tr")

        if cfg.local_rank == 0:

            if val_loss < best_val_loss:
                print(f'SAVING CHECKPOINT: val_loss {best_val_loss:.5} -> {val_loss:.5}')
                if cfg.local_rank == 0:

                    checkpoint = create_checkpoint(model, 
                                                optimizer, 
                                                epoch, 
                                                scheduler=scheduler, 
                                                scaler=scaler)

                    torch.save(checkpoint, f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_best_seed{cfg.seed}.pth")
                best_val_loss = val_loss

        if cfg.distributed:
            torch.distributed.barrier(group)


    writer.close()

    #END of training
    
    
    if cfg.local_rank == 0 and cfg.epochs > 0:
        print(f'SAVING LAST EPOCH: val_loss {val_loss:.5}')
        checkpoint = create_checkpoint(model, 
                                       optimizer, 
                                       epoch, 
                                       scheduler=scheduler, 
                                       scaler=scaler)

        torch.save(checkpoint, f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pth")
