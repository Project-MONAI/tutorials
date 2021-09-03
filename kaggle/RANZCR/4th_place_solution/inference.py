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
from scipy.stats import rankdata
from scipy.special import expit
from models.seg_model import RanzcrNet
from data.seg_data import CustomDataset
from torch.utils.data import DataLoader
import multiprocessing as mp
import glob

from monai.transforms import (
    Resized,
    SpatialPadd,
    CastToTyped,
    Compose,
    EnsureTyped,
    NormalizeIntensityd,
    Lambdad,
)


# In[3]:


COMP_FOLDER = '../input/ranzcr-clip-catheter-line-classification/'
N_CORES = mp.cpu_count()
test = pd.read_csv(f'{COMP_FOLDER}sample_submission.csv')
MP = True


# In[4]:


cfg = importlib.import_module('cfg_seg_40_1024d_full')
importlib.reload(cfg)
cfg = cfg.cfg

# test settings
cfg.data_folder = f'{COMP_FOLDER}test/'
cfg.data_dir = COMP_FOLDER
cfg.pretrained = False
cfg.device="cuda"

cfg.calc_loss = False

to_device_transform = ToDeviced(
    keys=("input", "target", "mask", "is_annotated"), device=cfg.device
)

print(cfg.backbone)


# In[5]:


state_dicts = []
for filepath in glob.iglob('../input/ranzcr-4th-place-reproduce-with-monai/weights/efnb8_ap/*.pth'):
    state_dicts.append(filepath)
    
state_dicts = state_dicts[:4]
    
print(state_dicts)

nets = []
for i in range(len(state_dicts)):
    d = torch.load(state_dicts[i])['model']
    new_d = {}
    for k,v in d.items():
        new_d[k.replace("module.", "")] = v
    sd = new_d
    
    net = RanzcrNet(cfg).eval().to(cfg.device)
    net.load_state_dict(sd)
    
    del net.decoder
    del net.segmentation_head
    
    nets.append(net)

test_augs = [
    Compose([
        Resized(keys=("input", "mask"), spatial_size=1008, size_mode="longest", mode="bilinear", align_corners=False),
        SpatialPadd(keys=("input", "mask"), spatial_size=(1008, 1008)),
        CastToTyped(keys="input", dtype=np.float32),
        NormalizeIntensityd(keys="input", nonzero=False),
        Lambdad(keys="input", func=lambda x: x.clip(-20, 20)),
        EnsureTyped(keys=("input", "mask")),
    ]),
]

if MP:
    cfg.batch_size = 32
else:
    cfg.batch_size = 16

with torch.no_grad():
    
    aug_preds = []
    for aug in test_augs:
        test_ds = CustomDataset(test, cfg, aug, mode="test")
        test_dl = DataLoader(test_ds, shuffle=False, batch_size = cfg.batch_size, num_workers = N_CORES)
    
        fold_preds = [[] for i in range(len(nets))]
        for batch in tqdm(test_dl):
            batch = to_device_transform(batch)
            for i, net in enumerate(nets):
                if MP:
                    with autocast():
                        logits = net(batch)['logits'].cpu().numpy()
                else:
                    logits = net(batch)['logits'].cpu().numpy()
                fold_preds[i] += [logits] 
        fold_preds = [np.concatenate(p) for p in fold_preds]
        
        aug_preds.append(fold_preds)

preds = np.stack(np.stack(aug_preds))
preds = preds.transpose(1,0,2,3)

preds = expit(preds)
preds = np.mean(preds, axis=0)
preds = rankdata(preds, axis=1) / preds.shape[1]
preds = np.mean(preds, axis=0)


# In[6]:


sub = test.copy()
sub[cfg.label_cols] = preds
submission = pd.read_csv(f'{COMP_FOLDER}sample_submission.csv')
submission.loc[sub.index, cfg.label_cols] = sub[cfg.label_cols]
submission.to_csv('submission.csv',index=False)

