from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import pandas as pd
from tqdm import tqdm
import ast
from monai.transforms import LoadImage


def batch_to_device(batch,device):
    batch_dict = {key:batch[key].to(device) for key in batch}
    return batch_dict

tr_collate_fn = None
val_collate_fn = None

class CustomDataset(Dataset):

    def __init__(self, df, cfg, aug, mode='train'):

        self.img_reader = LoadImage(image_only=True)
        self.cfg = cfg
        self.df = df.copy()
        self.annotated_df = pd.read_csv(cfg.data_dir + 'train_annotations.csv')
        annotated_ids = self.annotated_df['StudyInstanceUID'].unique()
        self.is_annotated = self.df['StudyInstanceUID'].isin(annotated_ids).astype(int).values
        self.fns = self.df['StudyInstanceUID'].values
        self.annotated_df = self.annotated_df.groupby('StudyInstanceUID')
        self.label_cols = np.array(cfg.label_cols)

        if mode == "train":
            self.annot = pd.read_csv(cfg.data_dir + "train_annotations.csv")
            self.annot = self.annot[self.annot.StudyInstanceUID.isin(self.df.StudyInstanceUID)]
        
        self.labels = self.df[self.label_cols].values
        self.mode = mode
        self.aug = aug
        self.data_folder = cfg.data_folder


    def get_thickness(self):

        if isinstance(self.cfg.thickness, list):
            thickness = np.random.randint(self.cfg.thickness[0], self.cfg.thickness[1])
        else:
            thickness = self.cfg.thickness
            
        return thickness

    def load_one(self, id_):
        ext = self.cfg.image_extension
        fp = self.data_folder + id_ + ext
        try:
            img = self.img_reader(filename=fp).transpose(1, 0)
            img = img[:,:,None]

        except:
            print("FAIL READING img", fp)
            img = np.zeros((self.img_size[0],self.img_size[1]), dtype=np.float32) 
           
        return img
    
    def get_mask(self,study_id,img_shape,is_annotated):

        if is_annotated == 0:
            return np.zeros((img_shape[0], img_shape[1], self.cfg.seg_dim))
        
        df = self.annotated_df.get_group(study_id)
        masks = []
        
        mask = np.zeros((img_shape[0], img_shape[1], self.cfg.seg_dim))
        #print(img_shape)
        
        #mask = mask.reshape(mask.shape[0], mask.shape[1],3)

        
        for idx, data in df.iterrows():

            xys = [np.array(ast.literal_eval(data['data'])).clip(0,np.inf).astype(np.int32)[:,None,:]]
            
            m = np.zeros(img_shape)

            #print(mask.shape)

            m = cv2.polylines(m, xys,False,1, thickness=self.get_thickness(), lineType =cv2.LINE_AA)
                #print(mask.sum())
            
            if self.cfg.seg_dim > 3:
                idx = np.where(self.label_cols == data["label"])[0][0]
            else:
                if "ETT" in data["label"] or self.cfg.seg_dim == 1:
                    idx = 0
                elif "NGT" in data["label"]:
                    idx = 1
                elif "CVC" in data["label"]:
                    idx = 2
                else:
                    continue
            
            #if 
            mask[:,:,idx][:,:,None] = np.max([mask[:,:,idx][:,:,None], m], axis=0)

        return mask

    def __getitem__(self, idx):
        
        fn = self.fns[idx]
        label = self.labels[idx]
        is_annotated = self.is_annotated[idx]
        
        img = self.load_one(fn)
        mask = self.get_mask(fn,img.shape, is_annotated).transpose(2, 0, 1)
        data = {"input": img.transpose(2, 0, 1), "mask": mask}
        if self.aug:
            data = self.aug(data)

        return {
            "input": data["input"],
            "target": torch.tensor(label).float(),
            "mask": data["mask"],
            "is_annotated": torch.tensor(is_annotated).float(),
        }
    
    def __len__(self):
        return len(self.fns)
