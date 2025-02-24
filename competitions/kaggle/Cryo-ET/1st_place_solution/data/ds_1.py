import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import zarr
from tqdm import tqdm

def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict

def collate_fn(batch):
    
    keys = batch[0].keys()
    batch_dict = {key:torch.cat([b[key] for b in batch]) for key in keys}
    return batch_dict

tr_collate_fn = collate_fn
val_collate_fn = collate_fn

import monai.data as md
import monai.transforms as mt


class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug, mode="train"):

        self.cfg = cfg
        self.mode = mode
        self.df = df
        self.experiment_df = self.df.drop_duplicates(subset='experiment').copy()
        self.exp_dict = self.df.groupby('experiment')
        self.class2id = {c:i for i,c in enumerate(cfg.classes)}
        self.n_classes = len(cfg.classes)
        self.data_folder = cfg.data_folder

            
        self.random_transforms = aug
        data = [self.load_one(img_id) for img_id in tqdm(self.experiment_df['experiment'].values)]
        data = md.CacheDataset(data=data, transform=cfg.static_transforms, cache_rate=1.0)
                
        if self.mode == 'train':
            self.monai_ds = md.Dataset(data=data, transform=self.random_transforms)
            self.sub_epochs = cfg.train_sub_epochs
            self.len = len(self.monai_ds) * self.sub_epochs
        else:
            self.monai_ds = md.CacheDataset(data=data, transform=self.random_transforms, cache_rate=1.0)[0]
            self.sub_epochs = cfg.val_sub_epochs
            self.len = len(self.monai_ds['image'])
            
    def __getitem__(self, idx):

        if self.mode =='train':
            monai_dict = self.monai_ds[idx//self.sub_epochs]
            feature_dict = {
                "input": torch.stack([item['image'] for item in monai_dict]),
                "target": torch.stack([item['label'] for item in monai_dict]),
            }  

        else:
            monai_dict = {k:self.monai_ds[k][idx] for k in self.monai_ds}
            monai_dict['location'] = torch.from_numpy(self.monai_ds['image'].meta['location'][:,idx])
            feature_dict = {
                "input": torch.stack([item['image'] for item in [monai_dict]]),
                "location": torch.stack([item['location'] for item in [monai_dict]]),
                "target": torch.stack([item['label'] for item in [monai_dict]]),
            }              
            
        return feature_dict

    def __len__(self):
        return self.len

    def load_one(self, experiment_id):
        

        img_fp = f'{self.data_folder}{experiment_id}'
        try:
            with zarr.open(img_fp + '/VoxelSpacing10.000/denoised.zarr') as zf:
                img = np.array(zf[0]).transpose(2,1,0)
            # img = np.array(zarr.open(img_fp + '/VoxelSpacing10.000/denoised.zarr')[0]).transpose(2,1,0)
        except Exception as e:
            print(e)
        
        centers = self.exp_dict.get_group(experiment_id)[['x','y','z']].values / 10
        classes = self.exp_dict.get_group(experiment_id)['particle_type'].map(self.class2id).values
        mask = np.zeros((self.n_classes,) + img.shape[-3:])
        mask[classes, centers[:,0].astype(int), centers[:,1].astype(int), centers[:,2].astype(int)] = 1
        return {'image':img, 'label':mask}



