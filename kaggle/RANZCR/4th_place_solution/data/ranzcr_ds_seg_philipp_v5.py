from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch 
import albumentations as A
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import pandas as pd
from tqdm import tqdm
import ast


def batch_to_device(batch,device):
    batch_dict = {key:batch[key].to(device) for key in batch}
    return batch_dict





tr_collate_fn = None
val_collate_fn = None

class CustomDataset(Dataset):

    def __init__(self, df, cfg, aug, mode='train'):

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
        self.normalization = cfg.normalization
        self.data_folder = cfg.data_folder
        self.cache_n_img = self.cfg.cache_n_img
        self.cached_img = 0

    def get_thickness(self):

        if isinstance(self.cfg.thickness, list):
            thickness = np.random.randint(self.cfg.thickness[0], self.cfg.thickness[1])
        else:
            thickness = self.cfg.thickness
            
        return thickness

    def load_one(self, id_):
        ext = self.cfg.image_extension
        if "png" in id_:
            ext = ""
        fp = self.data_folder + id_ + ext
        try:
            if self.cfg.rgb:
                img = cv2.imread(fp)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = cv2.imread(fp,cv2.IMREAD_UNCHANGED)
                img = img[:,:,None]
                if self.cfg.remove_border:
                    mask = img > 0
                    img = img[np.ix_(mask.any(1)[:,0], mask.any(0)[:,0])]
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
    
    

    
    def tubemix(self, img, label, is_annotated):
        
        #print("label", label)
        mask = None

        

        if is_annotated:
            if np.random.random() <= self.cfg.tubemix_proba:
                return img, label, mask
        
        curr_labels = self.label_cols[np.where(label==1)]

        if len(curr_labels) == 0:
            return img, label, mask

        curr_label = np.random.choice(curr_labels)
        #print(curr_labels)
         
        if np.random.random() <= 1:

            
            for i in range(10):
                #print(i)
                #rnd_idx = np.random.randint(len(self.annot))

                #rnd_idx = 8

                #rnd_idx = np.random.randint(len(self.annot))
                #annot = self.annot.iloc[rnd_idx]
                
                annot = self.annot
                
                annot = self.annot[self.annot.label == curr_label]
                
                # if pd.Series(curr_labels).str.contains("ETT").sum() > 0:
                #     annot = annot[~annot.label.isin(self.label_cols[:3])]
                
                annot = annot.sample(n=1, replace=False)
                
                rnd_img = self.load_one(annot.StudyInstanceUID.values[0])
                
                
                #print(annot.label.values[0])
                #print(img.shape, rnd_img.shape)

                if img.shape != rnd_img.shape:
                    continue

                #print(label)

                label[np.where(self.label_cols == annot.label.values[0])[0]] = 1
                #print(label)

                data = np.array(ast.literal_eval(annot["data"].values[0])).clip(0,np.inf).astype(np.int32)

                
                mask = np.zeros(img.shape)
                #'print(xys.shape)
                mask = cv2.polylines(mask,[data],False,1, thickness=self.get_thickness(), lineType =cv2.LINE_AA)

                img[np.where(mask==1)] = rnd_img[np.where(mask==1)]

                #print("yay")
                break
                
        #print("label", label)
            
        return img, label, mask

    #def tubedrop(self, img, label, is_annotated):

    def __getitem__(self, idx):
        
        fn = self.fns[idx]
        label = self.labels[idx]
        is_annotated = self.is_annotated[idx]
        
        if self.cache_n_img > 0:
            try:
                img = self.data_cache[fn] 
            except:
                img = self.load_one(fn)
                if self.cached_img < self.cache_n_img:
                    self.data_cache[fn] = img
                    self.cached_img += 1
        else:
            img = self.load_one(fn)
        
        mask = self.get_mask(fn,img.shape, is_annotated)

        if self.mode == "train":

           # if self.cfg.tubedrop:
           #     img, label, mask2 = self.tubedrop(img, label, is_annotated)

            if self.cfg.tubemix:
                img, label, mask2 = self.tubemix(img, label, is_annotated)

                if mask2 is not None:
                    mask[np.where(mask2==1)] = 1
                    is_annotated = 1

            if self.cfg.label_smoothing > 0:
                label[np.where(label==1)] = 1 - self.cfg.label_smoothing
                label[np.where(label==0)] = self.cfg.label_smoothing

        
        if self.aug:
            img, mask = self.augment(img, mask)
        
        img = img.astype(np.float32)
        if self.normalization:
            img = self.normalize_img(img)

    
        img_tensor = self.to_torch_tensor(img)
        mask_tensor = self.to_torch_tensor(mask)
        
        feature_dict = {'input':img_tensor,
                       'target':torch.tensor(label).float(),
                        'mask':mask_tensor,
                        'is_annotated':torch.tensor(is_annotated).float(),
                       }
        return feature_dict

    
    def __len__(self):
        return len(self.fns)


    
    
    def load_cache(self):
        for fn in tqdm(self.fns):
            self.data_cache[fn] = self.load_one(fn)

    def augment(self,img, mask):
        a = self.aug(image=img, mask=mask)
        img_aug = a['image']
        mask_aug = a['mask']
        return img_aug, mask_aug

    def normalize_img(self,img):
        
        if self.normalization == 'channel':
            pixel_mean = img.mean((0,1))
            pixel_std = img.std((0,1)) + 1e-6
            img = (img - pixel_mean[None,None,:]) / pixel_std[None,None,:]
            img = img.clip(-20,20)
           
        elif self.normalization == 'image':
            img = (img - img.mean()) / img.std() + 1e-6
            img = img.clip(-20,20)
            
        elif self.normalization == 'image_mean':
            img = (img - img.mean())

            
        elif self.normalization == 'simple':
            img = img/255
            
        elif self.normalization == 'inception':
            mean = np.array([0.5, 0.5 , 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5 , 0.5], dtype=np.float32)
            img = img.astype(np.float32)
            img = img/255.
            img -= mean
            img *= np.reciprocal(std, dtype=np.float32)
            
        elif self.normalization == 'imagenet':
            mean = np.array([123.675, 116.28 , 103.53 ], dtype=np.float32)
            std = np.array([58.395   , 57.120, 57.375   ], dtype=np.float32)
            img = img.astype(np.float32)
            img -= mean
            img *= np.reciprocal(std, dtype=np.float32)
            
        elif self.normalization == 'min_max':
            img = img - np.min(img)
            img = img / np.max(img)
            return img
        
        else:
            pass
        
        return img
    
    
    def to_torch_tensor(self,img):
        return torch.from_numpy(img.transpose((2, 0, 1)))
