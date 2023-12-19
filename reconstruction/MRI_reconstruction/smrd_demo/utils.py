import os
from collections import OrderedDict
from typing import Optional
from pathlib import Path

import numpy as np
import glob

import torch
import torch.fft as torch_fft
from torch import nn
import torchvision
import torch.distributed as dist

def dict2namespace(config: dict):
    '''
    Utility function to convert a dictionary to a namespace
    Args:
        config - dictionary
    
    Returns:
        namespace - argparse.Namespace object
    '''
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def get_sigmas(config):
    if config.model.sigma_dist == 'geometric':
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(config.model.sigma_begin), np.log(config.model.sigma_end),
                               config.model.num_classes))).float().to(config.device)
    elif config.model.sigma_dist == 'uniform':
        sigmas = torch.tensor(
            np.linspace(config.model.sigma_begin, config.model.sigma_end, config.model.num_classes)
        ).float().to(config.device)

    else:
        raise NotImplementedError('sigma distribution not supported')

    return sigmas

def np_ifft2(kspace):
    '''
    Perform ifft2 on numpy array.

    Args:
        kspace (np.ndarray): kspace data (..., H, W)
    Returns:
        output (np.ndarray): complex valued image data (..., H, W)
    '''
    output = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-1, -2)), axes=(-1, -2), norm='ortho'), axes=(-1, -2))
    return output



def scale(img):
    img = img.detach().cpu().numpy()
    magnitude_vals = np.abs(img).reshape(-1)
    if img.shape[0] == 320:
        k = int(round(0.015 * torch.from_numpy(magnitude_vals).numel()))
    else:
        k = int(round(0.02 * torch.from_numpy(magnitude_vals).numel()))
    scale = torch.min(torch.topk(torch.from_numpy(magnitude_vals), k).values)
    img = torch.clip(img / scale, 0, 1)
    return img

def normalize(gen_img, estimated_mvue):
    '''
        Estimate mvue from coils and normalize with 99% percentile.
    '''
    scaling = torch.quantile(estimated_mvue.abs(), 0.99)
    return gen_img * scaling

def unnormalize(gen_img, estimated_mvue):
    '''
        Estimate mvue from coils and normalize with 99% percentile.
    '''
    scaling = torch.quantile(estimated_mvue.abs(), 0.99)
    return gen_img / scaling

# Multicoil forward operator for MRI
class MulticoilForwardMRI(nn.Module):
    def __init__(self, orientation):
        self.orientation = orientation
        super(MulticoilForwardMRI, self).__init__()
        return

    # Centered, orthogonal ifft in torch >= 1.7
    def _ifft(self, x):
        x = torch_fft.ifftshift(x, dim=(-2, -1))
        x = torch_fft.ifft2(x, dim=(-2, -1), norm='ortho')
        x = torch_fft.fftshift(x, dim=(-2, -1))
        return x

    # Centered, orthogonal fft in torch >= 1.7
    def _fft(self, x):
        x = torch_fft.fftshift(x, dim=(-2, -1))
        x = torch_fft.fft2(x, dim=(-2, -1), norm='ortho')
        x = torch_fft.ifftshift(x, dim=(-2, -1))
        return x


    '''
    Inputs:
     - image = [B, H, W] torch.complex64/128    in image domain
     - maps  = [B, C, H, W] torch.complex64/128 in image domain
     - mask  = [B, W] torch.complex64/128 w/    binary values
    Outputs:
     - ksp_coils = [B, C, H, W] torch.complex64/128 in kspace domain
    '''
    def forward(self, image, maps, mask):
        # Broadcast pointwise multiply
        coils = image[:, None] * maps

        # Convert to k-space data
        ksp_coils = self._fft(coils)

        if self.orientation == 'vertical':
            # Mask k-space phase encode lines
            ksp_coils = ksp_coils * mask[:, None, None, :]
        elif self.orientation == 'horizontal':
            # Mask k-space frequency encode lines
            ksp_coils = ksp_coils * mask[:, None, :, None]
        else:
            if len(mask.shape) == 3:
                ksp_coils = ksp_coils * mask[:, None, :, :]
            else:
                raise NotImplementedError('mask orientation not supported')


        # Return downsampled k-space
        return ksp_coils

def get_mvue(kspace, s_maps):
    ''' Get mvue estimate from coil measurements '''
    ifft2_kspace = np_ifft2(kspace)
    return np.sum(ifft2_kspace * np.conj(s_maps), axis=1) / np.sqrt(np.sum(np.square(np.abs(s_maps)), axis=1))
    
def get_all_files(folder, pattern='*'):
    files = [x for x in glob.iglob(os.path.join(folder, pattern))]
    return sorted(files)

# Source: https://stackoverflow.com/questions/3229419/how-to-pretty-print-nested-dictionaries
def pretty(d, indent=0):
    ''' Print dictionary '''
    for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

def save_images(samples, loc, normalize=False):
    #convert loc to pathlib.Path
    loc = Path(loc)
    #get string representation of loc
    loc = str(loc)
    torchvision.utils.save_image(
        samples,
        loc,
        nrow=int(samples.shape[0] ** 0.5),
        normalize=normalize,
        scale_each=True,
        )

def load_dict(model, ckpt, device='cuda'):
    state_dict = torch.load(ckpt, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except:
        print('Loading model failed... Trying to remove the module from the keys...')
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_state_dict[key[len('module.'):]] = value
        model.load_state_dict(new_state_dict)
    return model

def update_pbar_desc(pbar, metrics, labels):
    pbar_string = ''
    for metric, label in zip(metrics, labels):
        if isinstance(metric, torch.Tensor):
            metric = metric.item()
        if isinstance(metric, float):
            pbar_string += f'{label}: {metric:.7f}; '
        else:
            pbar_string += f'{label}: {metric}; '
    pbar.set_description(pbar_string)
