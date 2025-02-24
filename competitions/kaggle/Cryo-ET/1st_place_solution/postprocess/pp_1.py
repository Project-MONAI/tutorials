

import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm

import torch
from torch import nn



def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool3d(x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    return torch.where(max_mask, scores, zeros)

def reconstruct(img, locations, out_size, crop_size):
    reconstructed_img = torch.zeros(out_size)
    
    for i in range(img.shape[0]):
        reconstructed_img[:,locations[0][i]:locations[0][i]+crop_size[0],
                          locations[1][i]:locations[1][i]+crop_size[1],
                          locations[2][i]:locations[2][i]+crop_size[2],
                         ] = img[i,:]
    return reconstructed_img


def post_process_pipeline(cfg, val_data, val_df):
    
    img = val_data['logits']
    img = torch.nn.functional.interpolate(img, size=(cfg.roi_size[0],cfg.roi_size[1],cfg.roi_size[2]), mode='trilinear', align_corners=False)
    locations = val_data['location']
    out_size = [cfg.n_classes + 1] + [l.item()+r for l,r in zip(locations.max(0)[0],cfg.roi_size)]
    rec_img = reconstruct(img, locations.permute(1,0), out_size=out_size, crop_size=cfg.roi_size)
    s = rec_img.shape[-3:]
    rec_img = torch.nn.functional.interpolate(rec_img[None], size=(s[0]//2,s[1]//2,s[2]//2), mode='trilinear', align_corners=False)[0]
    preds = rec_img.softmax(0)[:-1]
    
    pred_df = []

    for i,p in enumerate(cfg.classes):
        p1 = preds[i][None,].cuda()
        y = simple_nms(p1, nms_radius=int(0.5 * cfg.particle_radi[p]/10))
        kps = torch.where(y>0)
        xyz = torch.stack(kps[1:],-1) * 10 * 2
        conf = y[kps]
        pred_df_ = pd.DataFrame(xyz.cpu().numpy(),columns=['x','y','z'])
        pred_df_['particle_type'] = p
        pred_df_['conf'] = conf.cpu().numpy()
#         pred_df_['experiment'] = experiments[fold]
        pred_df += [pred_df_]
    pred_df = pd.concat(pred_df)
    pred_df = pred_df[(pred_df['x']<6300) & (pred_df['y']<6300)& (pred_df['z']<1840) & (pred_df['conf']>0.01)].copy()
    pred_df.to_csv(f"{cfg.output_dir}/fold{cfg.fold}/val_pred_df_seed{cfg.seed}.csv",index=False)
    return pred_df
