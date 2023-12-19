# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import multiprocessing
import argparse

from tqdm import tqdm

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from monai.networks.layers.conjugate_gradient import ConjugateGradient
from monai.losses.sure_loss import SURELoss, complex_diff_abs_loss

from models.ema import EMAHelper
from models.models import UNet
from utils import *

#need to install
from omegaconf import OmegaConf
import pickle, gzip
from matplotlib.pyplot import *

#--------------------------------------------------
#define the denoise and conjugate gradient step as one function
def denoise_cg_step(
    x, 
    score, 
    labels, 
    step_size, 
    noise, 
    lambda_t,
    y_cond,
    cg_solve_fn: ConjugateGradient,
    ):
    #denoise step
    with torch.no_grad():
        p_grad = score(x, labels)
    y = x + step_size * (p_grad) + noise
    #CG step
    cg_in = torch.view_as_complex((y_cond + lambda_t.clone() * y).permute(0,2,3,1)) 
    #solving eq.9 in the paper, using CG. with x0 = y_cond, y = y_cond + lambda_t * y.
    #during the CG steps, the update step will 'drag' the solution away from
    #y_cond, and towards y, based on the lambda_t value.
    y = cg_solve_fn(
        torch.view_as_complex(y_cond.permute(0,2,3,1)),
        cg_in,)
    y = torch.view_as_real(y).permute(0,-1,1,2)
    return y

class SMRDOptimizer(torch.nn.Module):
    def __init__(self, config, max_iter=2000, working_dir='./',  project_dir='./'):
        super().__init__()
        self.config = config
        self.SMRD_config = self._dict2namespace(config['langevin_config'])
        self.device = config['device']
        self.SMRD_config.device = config['device']
        self.project_dir = project_dir
        self.score = UNet(self.SMRD_config).to(self.device)
        self.sigmas_torch = get_sigmas(self.SMRD_config)
        self.sigmas = self.sigmas_torch.cpu().numpy()
        self.score = torch.nn.DataParallel(self.score)
        states = torch.load(os.path.join(project_dir, config['gen_ckpt']))
        import pdb; pdb.set_trace()
        self.score.load_state_dict(states[0], strict=True)
        self.max_iter = max_iter

        if self.SMRD_config.model.ema:
            ema_helper = EMAHelper(mu=self.SMRD_config.model.ema_rate)
            ema_helper.register(self.score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(self.score)
        del states
        self.index = 0
        self.working_dir = working_dir

    def _dict2namespace(self,SMRD_config):
        namespace = argparse.Namespace()
        for key, value in SMRD_config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    def _initialize(self):
        self.gen_outs = []

    def _ifft(self, x):
        x = torch_fft.ifftshift(x, dim=(-2, -1))
        x = torch_fft.ifft2(x, dim=(-2, -1), norm='ortho')
        x = torch_fft.fftshift(x, dim=(-2, -1))
        return x

    def _fft(self, x):
        x = torch_fft.fftshift(x, dim=(-2, -1))
        x = torch_fft.fft2(x, dim=(-2, -1), norm='ortho')
        x = torch_fft.ifftshift(x, dim=(-2, -1))
        return x

    def _sample(self, y):
        ref, mvue, maps, batch_mri_mask = y
        estimated_mvue = torch.tensor(
                get_mvue(ref.cpu().numpy(), maps.cpu().numpy()), device=ref.device)
        start_iter = self.config['start_iter']

        assert self.max_iter > start_iter, f'max_iter must be greater than start_iter, got {self.max_iter} and {start_iter}'
        # pbar = tqdm(range(self.SMRD_config.model.num_classes - start_iter), disable=(self.config['device'] != 0))
        pbar = tqdm(range(self.max_iter ), disable=(self.config['device'] != 0))

        pbar_labels = ['iter', 'step_size', 'error', 'mean', 'max']
        step_lr = self.SMRD_config.sampling.step_lr
        forward_operator = lambda x: MulticoilForwardMRI(
                self.config['orientation'])(torch.complex(x[:, 0], x[:, 1]), maps, batch_mri_mask)
        inverse_operator = lambda x: torch.view_as_real(
                torch.sum(self._ifft(x) * torch.conj(maps), axis=1)
                ).permute(0,3,1,2)
        samples = torch.rand(
                y[0].shape[0], 
                self.SMRD_config.data.channels, 
                self.config['image_size'][0], 
                self.config['image_size'][1], 
                device=self.device,
                )
        n_steps_each = 3
        window_size = self.config['window_size'] * n_steps_each 
        gt_losses = []
        lambdas = []
        SURES = []
        
        lamda_init = self.config.lambda_init
        lamda_end = self.config.lambda_end
        if self.config.lambda_func == 'cnst':
            lamdas = lamda_init * torch.ones(len(self.sigmas), device=samples.device)
        elif self.config.lambda_func == 'linear':
            lamdas = torch.linspace(lamda_init, lamda_end, len(self.sigmas))
        elif self.config.lambda_func == 'learnable':
            with torch.enable_grad():
                init = torch.tensor(lamda_init, dtype=torch.float32)
                lamda = torch.nn.Parameter(init, requires_grad=True)
            lambda_lr = self.config['lambda_lr']
            optimizer = torch.optim.Adam([lamda], lr=lambda_lr)            
        
        with torch.no_grad():
            for c in pbar:
                c = c + start_iter
                if c <= 1800:
                    n_steps_each = 3
                else:
                    n_steps_each = self.SMRD_config.sampling.n_steps_each
                sigma = self.sigmas[c]
                labels = torch.ones(samples.shape[0], device=samples.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / self.sigmas[-1]) ** 2

                for s in range(n_steps_each):
                    with torch.enable_grad():
                        if self.config.lambda_func == 'learnable':
                            optimizer.zero_grad()
                        samples = samples.to('cuda')
                        if self.config.lambda_func == 'learnable':
                            samples = samples.requires_grad_(True)
                        noise = torch.randn_like(samples) * np.sqrt(step_size * 2)
                        # get score from model
                        with torch.no_grad():
                            p_grad = self.score(samples, labels)
                        estimated_mvue = estimated_mvue.clone().to('cuda')

                        if self.config.lambda_func == 'learnable':
                            samples = samples.requires_grad_(True)
                        
                        torch.autograd.set_detect_anomaly(True)
                        if self.config.lambda_func == 'learnable':
                            pass
                        else:
                            lamda = lamdas[c]
                            
                        if lamda.detach().cpu().numpy() > 0:
                            lamda_applied = lamda.clone()
                        else:
                            #If learning results in a negative lamda, apply initial lamda
                            lamda_applied = torch.tensor(lamda_init, dtype=torch.float32)
                        
                        #define the linear operator, in this case a 2D IFFT
                        model_normal = lambda m: torch.view_as_complex((unnormalize(inverse_operator(forward_operator(normalize(torch.view_as_real(m).permute(0,-1,1,2),estimated_mvue))),estimated_mvue) + lamda_applied.clone() * torch.view_as_real(m).permute(0,-1,1,2)).permute(0,2,3,1))
                        cg_solve = ConjugateGradient(model_normal, self.config['num_cg_iter'])
                        n = samples.size(0)
                        meas = forward_operator(samples) #H x hat t, ref = y
                        zf = inverse_operator(ref)
                        zf = unnormalize(zf, estimated_mvue)
                        zf = zf.type(torch.float32)
                        samples_input = samples
                        samples = samples.to(self.device)
                        
                        ### REVERSE DIFFUSION (aka denoise) ###
                        #Line 3 in Algo.1 in the paper
                        samples = samples + step_size * (p_grad) + noise 
                        #
                        
                        #Line 4
                        cg_in = torch.view_as_complex((zf + lamda_applied.clone() * samples).permute(0,2,3,1))
                        samples = cg_solve(torch.view_as_complex(zf.permute(0,2,3,1)),cg_in)
                        #

                        samples = torch.view_as_real(samples).permute(0,-1,1,2).type(torch.FloatTensor)
                        if self.config.lambda_func == 'learnable':
                            samples = samples.requires_grad_(True)
                        samples = samples.to(self.device)

                        # compute metrics
                        metrics = [c-start_iter, step_size, (meas-ref).norm(), (p_grad).abs().mean(), (p_grad).abs().max()]
                        update_pbar_desc(pbar, metrics, pbar_labels)
                        
                        # create perturbed input for monte-carlo SURE
                        #Line 5-6--
                        # eps = torch.abs(zf.max())/1000
                        # perturb_noise = torch.randn_like(samples) #\mu in algo1 line 5 he paper
                        SureLoss = SURELoss(
                                perturb_noise=torch.randn_like(samples),
                                eps=torch.abs(zf.max())/1000,)

                        #operator is a partial function of denoise_cg_step
                        denoise_cg_operator = lambda x: denoise_cg_step(
                            x, 
                            score=self.score, 
                            labels=labels, 
                            step_size=step_size, 
                            noise=noise, 
                            lambda_t=lamda_applied.clone(),
                            y_cond=zf,
                            cg_solve_fn=cg_solve,
                            )

                        #apply the SURE loss function
                        sure_loss = SureLoss(
                                operator=denoise_cg_operator,
                                x=samples_input,
                                y_pseudo_gt=zf,
                                y_ref=samples,
                                complex_input=True,
                                )
                        SURE = sure_loss
                        #--Line 5-6

                        SURES.append(SURE.detach().cpu().numpy())
                        gt_l2_loss = complex_diff_abs_loss(samples, mvue.squeeze(1))
                        gt_losses.append(gt_l2_loss.detach().cpu().numpy())
                        lambdas.append(lamda.clone().detach().cpu().numpy())

                        init_lambda_update = self.config['init_lambda_update']
                        last_lambda_update = self.config['last_lambda_update']
                        if c>init_lambda_update:
                            if c<last_lambda_update:
                                if self.config.lambda_func == 'learnable':
                                    #we will use SURE loss to update lambda
                                    loss = SURE
                                    loss.backward(retain_graph=True)
                                    optimizer.step()
                                
                        if self.config.lambda_func == 'learnable':
                            samples = samples.detach()
                            lamda = lamda.detach()
                            zf = zf.detach()
                            loss = loss.detach()
                    
                    if self.config.early_stop == 'stop':
                        # EARLY STOPPING USING SURE LOSS
                        # check the self-validation loss for early stopping
                        if len(SURES) > 3 * window_size:
                            if c > 3*window_size:
                                if np.mean(SURES[-window_size:]) > np.mean(SURES[-2*window_size:-window_size]): 
                                    print('\nAutomatic early stopping activated.')
                                    print(f'total iter at early_stop ={c-start_iter}')
                                    outputs = normalize(samples, estimated_mvue)
                                    outputs = outputs.permute(0,2,3,1)
                                    outputs = outputs.contiguous()
                                    outputs = torch.view_as_complex(outputs)
                                    norm_output = torch.abs(outputs).cpu().numpy()
                                    gt = torch.abs(mvue).squeeze(1).cpu().numpy()
                                    img_gen = normalize(samples, estimated_mvue)
                                    to_display = torch.view_as_complex(img_gen.permute(0, 2, 3, 1).reshape(-1, self.config['image_size'][0], self.config['image_size'][1], 2).contiguous()).abs()
                                    if self.config['anatomy'] == 'brain':
                                        # flip vertically
                                        to_display = to_display.flip(-2)
                                    elif self.config['anatomy'] == 'knees':
                                        # flip vertically and horizontally
                                        to_display = to_display.flip(-2)
                                        #to_display = to_display.flip(-1)
                                    elif self.config['anatomy'] == 'stanford_knees':
                                        # do nothing
                                        pass
                                    elif self.config['anatomy'] == 'abdomen':
                                        # flip horizontally
                                        to_display = to_display.flip(-1)
                                    else:
                                        pass
                                    to_display = scale(to_display)
                                    for i, exp_name in enumerate(self.config['exp_names']):
                                        file_name = f'{self.working_dir}/{exp_name}_ES_recon_R={self.config["R"]}_iter={c-start_iter:04d}.jpg'
                                        save_images(to_display[i:i+1], file_name, normalize=True)
                                    return normalize(samples, estimated_mvue)
                    else:
                        pass

                    if np.isnan((meas - ref).norm().cpu().numpy()):
                        return normalize(samples, estimated_mvue)

                if self.config['save_images']:
                    if (c) % self.config['save_iter'] ==0:
                        estimated_mvue = estimated_mvue.cpu()
                        img_gen = normalize(samples, estimated_mvue)
                        outputs = normalize(samples, estimated_mvue)
                        outputs = outputs.permute(0,2,3,1)
                        outputs = outputs.contiguous()
                        outputs = torch.view_as_complex(outputs)
                        norm_output = torch.abs(outputs).cpu().numpy()
                        gt = torch.abs(mvue).squeeze(1).cpu().numpy()
                        
                        to_display = torch.view_as_complex( img_gen.permute(0, 2, 3, 1).reshape(-1, self.config['image_size'][0], self.config['image_size'][1], 2).contiguous()).abs()
                        if self.config['anatomy'] == 'brain':
                            # flip vertically
                            to_display = to_display.flip(-2)
                        elif self.config['anatomy'] == 'stanford_knees':
                            # do nothing
                            pass
                        else:
                            pass

                        to_display = scale(to_display)
                        for i, exp_name in enumerate(self.config['exp_names']):
                            file_name = f'{self.working_dir}/{exp_name}_R={self.config["R"]}_iter={c-start_iter:04d}.jpg'
                            save_images(to_display[i:i+1], file_name, normalize=True)

        samples = samples.detach()
        return normalize(samples, estimated_mvue)

    def sample(self, y):
        self._initialize()
        mvue = self._sample(y)
        outputs = []
        for i in range(y[0].shape[0]):
            outputs_ = {
                'mvue': mvue[i:i+1],
            }
            outputs.append(outputs_)
        return outputs
    
def run(rank, config, project_dir, working_dir):
    # setup
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    project_name = config['anatomy']
    # pretty(config)
    config['device'] = rank
    sampler = None
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    SMRD_optimizer = SMRDOptimizer(config, max_iter=2000, working_dir=working_dir, project_dir=project_dir)
    SMRD_optimizer.to(rank)

    # load data
    with gzip.open('data/demo_data.pkl.gz', 'rb') as f: 
        sample = pickle.load(f)

    '''
    ref: one complex image per coil
    mvue: one complex image reconstructed using the coil images and the sensitivity maps
    maps: sensitivity maps for each one of the coils
    mask: binary valued kspace mask
    '''
    ref, mvue, maps, mask = sample['ground_truth'], sample['mvue'], sample['maps'], sample['mask']
    mvue, maps, mask = torch.from_numpy(mvue), torch.from_numpy(maps), torch.from_numpy(mask) #from numpy to torch
    ref, mvue, maps, mask = ref.unsqueeze(0), mvue.unsqueeze(0), maps.unsqueeze(0), mask.unsqueeze(0) #add batch dimension

    # move everything to cuda
    ref = ref.to(rank).type(torch.complex128)
    mask = mask.to(rank)
    noise_std = config['noise_std']
    noise = mask[None, :, :] * torch.view_as_complex(torch.randn(ref.shape+(2,)).to(rank)) * noise_std * torch.abs(ref).max()

    ref = ref + noise.to(rank)
    mvue = mvue.to(rank)
    maps = maps.to(rank)
    
    estimated_mvue = torch.tensor(
        get_mvue(ref.cpu().numpy(),
        maps.cpu().numpy()), 
        device=ref.device,
        )

    exp_names = []

    batch_idx = 0
    exp_name = 'demo'
    exp_names.append(exp_name)

    # save images for initial estimation and ground truth
    if config['save_images']:
        file_name = f'{working_dir}/{exp_name}_R={config["R"]}_ZF.jpg'
        estimated_mvuevis = torch.abs(estimated_mvue)
        estimated_mvuevis = scale(estimated_mvuevis)
        save_images(estimated_mvuevis[batch_idx:batch_idx+1].abs().flip(-2), file_name, normalize=True)
        mvuevis = torch.abs(mvue)
        mvuevis = scale(mvuevis)
        file_name = f'{working_dir}/{exp_name}_GT.jpg'
        save_images(mvuevis[batch_idx:batch_idx+1].abs().flip(-2), file_name, normalize=True)

    SMRD_optimizer.config['exp_names'] = exp_names
    SMRD_optimizer.slice_id = 0

    # run sampling
    outputs = SMRD_optimizer.sample((ref, mvue, maps, mask))

    # save results
    outputs[0] = outputs[0]['mvue'].permute(0,2,3,1)
    outputs[0] = torch.view_as_complex(outputs[0])
    norm_output = torch.abs(outputs[0]).cpu().numpy()
    gt = torch.abs(mvue).squeeze(1).cpu().numpy()
    img = scale(torch.from_numpy(norm_output))
    file_name = f'{working_dir}/{exp_name}_final_recon.jpg'
    save_images(img[batch_idx:batch_idx+1].abs().flip(-2), file_name, normalize=True)

def main():
    """ setup """
    #load config file
    config = OmegaConf.load('configs/demo/SMRD-brain_T2-noise005-R8.yaml')
    #get output folder
    working_dir = 'demo-outputs/'
    #make output folder
    os.makedirs(working_dir, exist_ok=True)
    #get current working directory
    project_dir = './'
    #run
    run(rank=0, config=config, project_dir=project_dir, working_dir=working_dir, )

if __name__ == '__main__':
    main()
