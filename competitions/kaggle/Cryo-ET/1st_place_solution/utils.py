
import random
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler, RandomSampler, SequentialSampler, DataLoader, WeightedRandomSampler
from torch import nn, optim
# from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import lr_scheduler
import importlib
import math
import neptune
from neptune.utils import stringify_unsupported

import logging
import pickle



def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    from https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles= 0.5, last_epoch= -1):
    """
    from https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)




def calc_grad_norm(parameters,norm_type=2.):
    
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    if torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        total_norm = None
        
    return total_norm

def calc_weight_norm(parameters,norm_type=2.):

    # l2_loss = 0
    # for param in parameters :
    #     l2_loss += 0.5 * torch.sum(param ** 2)
    # return l2_loss
    
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device


    total_norm = torch.stack([torch.norm(p.detach(), norm_type).to(device) for p in parameters]).mean()
    if torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        total_norm = None
        
    return total_norm

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
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[
            self.rank * self.num_samples : self.rank * self.num_samples + self.num_samples
        ]
        print(
            "SAMPLES",
            self.rank * self.num_samples,
            self.rank * self.num_samples + self.num_samples,
        )
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def sync_across_gpus(t, world_size):
    torch.distributed.barrier()
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


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_model(cfg, ds):
    Net = importlib.import_module(cfg.model).Net
    net = Net(cfg)
    if cfg.pretrained_weights is not None:
        if type(cfg.pretrained_weights) == list:
            cfg.pretrained_weights = cfg.pretrained_weights[cfg.fold]
        print(f'{cfg.local_rank}: loading weights from',cfg.pretrained_weights)
        state_dict = torch.load(cfg.pretrained_weights, map_location='cpu')
        if "model" in state_dict.keys():
            state_dict = state_dict['model']
        state_dict = {key.replace('module.',''):val for key,val in state_dict.items()}
        if cfg.pop_weights is not None:
            print(f'popping {cfg.pop_weights}')
            to_pop = []
            for key in state_dict:
                for item in cfg.pop_weights:
                    if item in key:
                        to_pop += [key]
            for key in to_pop:
                print(f'popping {key}')
                state_dict.pop(key)

        net.load_state_dict(state_dict, strict=cfg.pretrained_weights_strict)
        print(f'{cfg.local_rank}: weights loaded from',cfg.pretrained_weights)
    
    return net


def create_checkpoint(cfg, model, optimizer, epoch, scheduler=None, scaler=None):

    
    state_dict = model.state_dict()
    if cfg.save_weights_only:
        checkpoint = {"model": state_dict}
        return checkpoint
    
    checkpoint = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint

def load_checkpoint(cfg, model, optimizer, scheduler=None, scaler=None):
    
    print(f'loading ckpt {cfg.resume_from}')
    checkpoint = torch.load(cfg.resume_from, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler_dict = checkpoint['scheduler']
    if scaler is not None:    
        scaler.load_state_dict(checkpoint['scaler'])
        
    epoch = checkpoint['epoch']
    return model, optimizer, scheduler_dict, scaler, epoch


def get_dataset(df, cfg, mode='train'):
    
    #modes train, val, index
    print(f"Loading {mode} dataset")

    if mode == 'train':
        dataset = get_train_dataset(df, cfg)
#     elif mode == 'train_val':
#         dataset = get_val_dataset(df, cfg)
    elif mode == 'val':
        dataset = get_val_dataset(df, cfg)
    elif mode == 'test':
        dataset = get_test_dataset(df, cfg)
    else:
        pass
    return dataset

def get_dataloader(ds, cfg, mode='train'):
    
    if mode == 'train':
        dl = get_train_dataloader(ds, cfg)
    elif mode =='val':
        dl = get_val_dataloader(ds, cfg)
    elif mode =='test':
        dl = get_test_dataloader(ds, cfg)
    return dl


def get_train_dataset(train_df, cfg):

    train_dataset = cfg.CustomDataset(train_df, cfg, aug=cfg.train_aug, mode="train")
    if cfg.data_sample > 0:
        train_dataset = torch.utils.data.Subset(train_dataset, np.arange(cfg.data_sample))
    return train_dataset




def get_train_dataloader(train_ds, cfg):

    if cfg.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=cfg.world_size, rank=cfg.local_rank, shuffle=True, seed=cfg.seed
        )
    else:
        try:
            if cfg.random_sampler_frac > 0:
                
                num_samples = int(len(train_ds) * cfg.random_sampler_frac)
                sample_weights = train_ds.sample_weights
                sampler = WeightedRandomSampler(sample_weights, num_samples= num_samples )
            else:
                sampler = None
        except:
            sampler = None
            
            
    if cfg.use_custom_batch_sampler:
        sampler = RandomSampler(train_ds)
        bsampler = CustomBatchSampler(sampler, batch_size =cfg.batch_size, drop_last=cfg.drop_last)
        train_dataloader = DataLoader(
            train_ds,
            batch_sampler=bsampler,
#             shuffle=(sampler is None),
#             batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            collate_fn=cfg.tr_collate_fn,
#             drop_last=cfg.drop_last,
            worker_init_fn=worker_init_fn,
        )
    else:
        

        train_dataloader = DataLoader(
            train_ds,
            sampler=sampler,
            shuffle=(sampler is None),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            collate_fn=cfg.tr_collate_fn,
            drop_last=cfg.drop_last,
            worker_init_fn=worker_init_fn,
        )
    print(f"train: dataset {len(train_ds)}, dataloader {len(train_dataloader)}")
    return train_dataloader


def get_val_dataset(val_df, cfg, allowed_targets=None):
    val_dataset = cfg.CustomDataset(val_df, cfg, aug=cfg.val_aug, mode="val")
    return val_dataset


def get_val_dataloader(val_ds, cfg):

    if cfg.distributed and cfg.eval_ddp:
        sampler = OrderedDistributedSampler(
            val_ds, num_replicas=cfg.world_size, rank=cfg.local_rank
        )
    else:
        sampler = SequentialSampler(val_ds)

    if cfg.batch_size_val is not None:
        batch_size = cfg.batch_size_val
    else:
        batch_size = cfg.batch_size
    val_dataloader = DataLoader(
        val_ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=cfg.val_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    print(f"valid: dataset {len(val_ds)}, dataloader {len(val_dataloader)}")
    return val_dataloader


def get_test_dataset(test_df, cfg):
    test_dataset = cfg.CustomDataset(test_df, cfg, aug=cfg.val_aug, mode="test")
    return test_dataset


def get_test_dataloader(test_ds, cfg):

    if cfg.distributed and cfg.eval_ddp:
        sampler = OrderedDistributedSampler(
            test_ds, num_replicas=cfg.world_size, rank=cfg.local_rank
        )
    else:
        sampler = SequentialSampler(test_ds)

    if cfg.batch_size_val is not None:
        batch_size = cfg.batch_size_val
    else:
        batch_size = cfg.batch_size
    test_dataloader = DataLoader(
        test_ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=cfg.val_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    print(f"test: dataset {len(test_ds)}, dataloader {len(test_dataloader)}")
    return test_dataloader



def get_optimizer(model, cfg):

    params = model.parameters()

    if cfg.optimizer == "Adam":
        optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
   
    elif cfg.optimizer == "AdamW_plus":
        paras = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        params = [{"params": [param for name, param in paras if (not any(nd in name for nd in no_decay))],
                   "lr": cfg.lr,
                   "weight_decay":cfg.weight_decay},
                  {"params": [param for name, param in paras if (any(nd in name for nd in no_decay))],
                   "lr": cfg.lr,
                   "weight_decay":0.},
                 ]        
        optimizer = optim.AdamW(params, lr=cfg.lr)         

        
    elif cfg.optimizer == "AdamW":
        optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        
    elif cfg.optimizer == "SGD":
        optimizer = optim.SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.sgd_momentum,
            nesterov=cfg.sgd_nesterov,
            weight_decay=cfg.weight_decay,
        )

    return optimizer



def get_scheduler(cfg, optimizer, total_steps):

    if cfg.schedule == "steplr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.epochs_step * (total_steps // cfg.batch_size) // cfg.world_size,
            gamma=0.5,
        )
    elif cfg.schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup * (total_steps // cfg.batch_size) // cfg.world_size,
            num_training_steps=cfg.epochs * (total_steps // cfg.batch_size) // cfg.world_size,
            num_cycles = cfg.num_cycles
        )
    elif cfg.schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=cfg.epochs * (total_steps // cfg.batch_size) // cfg.world_size,
        )
        
    elif cfg.schedule == "CosineAnnealingLR":
        T_max = int(np.ceil(0.5*total_steps))
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=T_max, 
                                                   eta_min=1e-8)

#         print("num_steps", (total_steps // cfg.batch_size) // cfg.world_size)

        
        
    else:
        scheduler = None

    return scheduler


def setup_neptune(cfg):
    neptune_run = None
    if cfg.neptune_project:
        neptune_run = neptune.init_run(
            project=cfg.neptune_project,
#             tags=cfg.tags,
            mode=cfg.neptune_connection_mode,
            capture_stdout=False,
            capture_stderr=False,
            source_files=[f'models/{cfg.model}.py',f'data/{cfg.dataset}.py',f'configs/{cfg.name}.py']
        )


        neptune_run["cfg"] = stringify_unsupported(cfg.__dict__)

    return neptune_run


def read_df(fn):

    if 'parquet' in fn:
        df = pd.read_parquet(fn, engine = "fastparquet")
    else:
        df = pd.read_csv(fn)
    return df

def get_data(cfg):

    # setup dataset
    if type(cfg.train_df) == list:
        cfg.train_df = cfg.train_df[cfg.fold]
    print(f"reading {cfg.train_df}")
    df = read_df(cfg.train_df)

    if cfg.test:
        test_df = read_df(cfg.test_df)
    else:
        test_df = None
    
    if cfg.val_df:
        if type(cfg.val_df) == list:
            cfg.val_df = cfg.val_df[cfg.fold]
        val_df = read_df(cfg.val_df)
        if cfg.fold > -1:
            if 'fold' in val_df.columns:
                val_df = val_df[val_df["fold"] == cfg.fold]  
                train_df = df[df["fold"] != cfg.fold]   
            else:
                train_df = df             
        else:
            train_df = df
    else:
        if cfg.fold == -1:
            val_df = df[df["fold"] == 0]
        else:
            val_df = df[df["fold"] == cfg.fold]    
        
        train_df = df[df["fold"] != cfg.fold]
        
    return train_df, val_df, test_df


def upload_s3(cfg):
    from boto3.session import Session
    import boto3
    BUCKET_NAME = cfg.s3_bucket_name
    ACCESS_KEY = cfg.s3_access_key
    SECRET_KEY = cfg.s3_secret_key
    session = Session(aws_access_key_id=ACCESS_KEY,
                aws_secret_access_key=SECRET_KEY)
    s3 = session.resource('s3')

    s3.Bucket(BUCKET_NAME).upload_file(f"{cfg.output_dir}/fold{cfg.fold}/val_data_seed{cfg.seed}.pth", f"output/{cfg.name}/fold{cfg.fold}/val_data_seed{cfg.seed}.pth")
    s3.Bucket(BUCKET_NAME).upload_file(f"{cfg.output_dir}/fold{cfg.fold}/test_data_seed{cfg.seed}.pth", f"output/{cfg.name}/fold{cfg.fold}/test_data_seed{cfg.seed}.pth")
    s3.Bucket(BUCKET_NAME).upload_file(f"{cfg.output_dir}/fold{cfg.fold}/submission_seed{cfg.seed}.csv", f"output/{cfg.name}/fold{cfg.fold}/submission_seed{cfg.seed}.csv")


def flatten(t):
    return [item for sublist in t for item in sublist]

def set_pandas_display():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows',10000)
    pd.set_option('display.width', 10000)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

def dumpobj(file, obj):
    with open(file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def loadobj(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

def get_level(level_str):
    ''' get level'''
    l_names = {logging.getLevelName(lvl).lower(): lvl for lvl in [10, 20, 30, 40, 50]} # noqa
    return l_names.get(level_str.lower(), logging.INFO)

def get_logger(name, level_str):
    ''' get logger'''
    logger = logging.getLogger(name)
    logger.setLevel(get_level(level_str))
    handler = logging.StreamHandler()
    handler.setLevel(level_str)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')) # pylint: disable=C0301 # noqa
    logger.addHandler(handler)

    return logger

