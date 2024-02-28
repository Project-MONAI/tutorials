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
import argparse
import numpy as np
import torch
import torch.fft as torch_fft

from tqdm import tqdm

from monai.networks.layers.conjugate_gradient import ConjugateGradient
from monai.losses.sure_loss import SURELoss, complex_diff_abs_loss

from models.ema import EMAHelper
from mutils import (
    ifft,
    normalize,
    unnormalize,
    scale,
    get_sigmas,
    dict2namespace,
    get_mvue,
    update_pbar_desc,
    MulticoilForwardMRI,
)

import matplotlib.pyplot as plt


def _dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def denoise_cg_step(
    x,
    score,
    labels,
    step_size,
    noise,
    lambda_t,
    x_zf,
    cg_solve_fn: ConjugateGradient,
):
    # denoise step function with conjugate gradient
    # the output of this function is the denoised image, corresponding
    # to x_{t+1} = h(x_t, \lambda_t) in line 4 of Algo.1 in the paper.
    with torch.no_grad():
        p_grad = score(x, labels)
    x_update = x + step_size * (p_grad) + noise
    # CG step
    # solving eq.9 in the paper, using CG.
    # during the CG steps, the update step will 'drag' the solution away from
    # x_zf, and towards x_update, based on the lambda_t value.

    # More specifically, solve the equation (A^H A + lambda_t I) x = x_zf + lambda_t * x_update
    # where A is the linear operator (in this case, 2D FFT), and x is the reconstructed image
    x_update = cg_solve_fn(
        x=torch.view_as_complex(x_zf.permute(0, 2, 3, 1)),
        y=torch.view_as_complex((x_zf + lambda_t.clone() * x_update).permute(0, 2, 3, 1)),
    )
    x_update = torch.view_as_real(x_update).permute(0, -1, 1, 2)
    return x_update


class SMRDOptimizer(torch.nn.Module):
    def __init__(self, config, UNet, project_dir="./"):
        super().__init__()
        self.config = config
        self.SMRD_config = _dict2namespace(config["langevin_config"])
        self.device = config["device"]
        self.SMRD_config.device = config["device"]
        self.project_dir = project_dir
        self.score = UNet
        self.sigmas_torch = get_sigmas(self.SMRD_config)
        self.sigmas = self.sigmas_torch.cpu().numpy()
        self.score = torch.nn.DataParallel(self.score)
        states = torch.load(os.path.join(project_dir, config["gen_ckpt"]))
        self.score.load_state_dict(states[0], strict=True)
        if self.SMRD_config.model.ema:
            ema_helper = EMAHelper(mu=self.SMRD_config.model.ema_rate)
            ema_helper.register(self.score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(self.score)
        del states
        self.index = 0
        self.multicoil_forward_mri = MulticoilForwardMRI(self.config["orientation"])

    def _initialize(self):
        self.gen_outs = []

    def _sample(self, y):
        ref, mvue, maps, batch_mri_mask = y
        estimated_mvue = torch.tensor(get_mvue(ref.cpu().numpy(), maps.cpu().numpy()), device=ref.device)

        pbar = tqdm(
            range(self.config["start_iter"] + 1, self.SMRD_config.model.num_classes),
            disable=(self.config["device"] != 0),
        )
        pbar_labels = ["step_size", "error", "mean", "max"]
        step_lr = self.SMRD_config.sampling.step_lr

        def forward_operator(x):
            return self.multicoil_forward_mri(torch.complex(x[:, 0], x[:, 1]), maps, batch_mri_mask)

        def inverse_operator(x):
            return torch.view_as_real(torch.sum(ifft(x) * torch.conj(maps), axis=1)).permute(0, 3, 1, 2)

        samples = torch.rand(
            y[0].shape[0],
            self.SMRD_config.data.channels,
            self.config["image_size"][0],
            self.config["image_size"][1],
            device=self.device,
        )
        n_steps_each = 3
        window_size = self.config["window_size"] * n_steps_each
        gt_losses = []
        lambdas = []
        sures = []

        lamda_init = self.config.lambda_init
        lamda_end = self.config.lambda_end
        if self.config.lambda_func == "cnst":
            lamdas = lamda_init * torch.ones(len(self.sigmas), device=samples.device)
        elif self.config.lambda_func == "linear":
            lamdas = torch.linspace(lamda_init, lamda_end, len(self.sigmas))
        elif self.config.lambda_func == "learnable":
            with torch.enable_grad():
                init = torch.tensor(lamda_init, dtype=torch.float32)
                lamda = torch.nn.Parameter(init, requires_grad=True)
            lambda_lr = self.config["lambda_lr"]
            optimizer = torch.optim.Adam([lamda], lr=lambda_lr)

        with torch.no_grad():
            for c in pbar:
                if c <= 1800:
                    n_steps_each = 3
                else:
                    n_steps_each = self.SMRD_config.sampling.n_steps_each
                sigma = self.sigmas[c]
                labels = torch.ones(samples.shape[0], device=samples.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / self.sigmas[-1]) ** 2

                for _ in range(n_steps_each):
                    with torch.enable_grad():
                        if self.config.lambda_func == "learnable":
                            optimizer.zero_grad()
                        samples = samples.to("cuda")
                        if self.config.lambda_func == "learnable":
                            samples = samples.requires_grad_(True)
                        noise = torch.randn_like(samples) * np.sqrt(step_size * 2)
                        # get score from model
                        with torch.no_grad():
                            p_grad = self.score(samples, labels)
                        estimated_mvue = estimated_mvue.clone().to("cuda")

                        if self.config.lambda_func == "learnable":
                            samples = samples.requires_grad_(True)

                        torch.autograd.set_detect_anomaly(True)
                        if self.config.lambda_func == "learnable":
                            pass
                        else:
                            lamda = lamdas[c]

                        if lamda.detach().cpu().numpy() > 0:
                            lamda_applied = lamda.clone()
                        else:
                            # If learning results in a negative lamda, apply initial lamda
                            lamda_applied = torch.tensor(lamda_init, dtype=torch.float32)

                        # Below, we will define the linear operator, theorectically, we can
                        # write it as a matrix, and apply it to the input: Ax = y
                        # where A could be a concatenation of several linear oprations (eg. 2D FFT),
                        # x is the input, and y is the output.
                        # However, in practice, we don't need to compute the matrix A, since usually
                        # the dimension of the matrix is too large to be stored in memory, and it's
                        # not efficient to compute the inverse of A. Instead, we can use the
                        # conjugate gradient method to solve the equation Ax = y, without explicitly
                        # computing the matrix A. The conjugate gradient method is an iterative method to solve the equation

                        # The linear operator here is the solution to the following optimization problem:
                        # min_x ||Ax - y||^2 + lambda_t * ||x - x_update ||^2
                        # where A is a 2D FFT, y is the given measurement in the spectral domain,
                        # x_update is the 'hallucinated' image from the generative model, and x is the
                        # reconstructed image in the spatial domain. lambda_t is a hyperparameter that controls
                        # the trade-off between the data fidelity (first) term and the regularization (second) term.
                        # The solution to the above optimization problem is given by the following equation:
                        # x = (A^H A + lambda_t I)^-1 (A^H y + lambda_t x_update)
                        # where A^H is the conjugate transpose of A, and I is the identity matrix.
                        # The above equation can be solved using the conjugate gradient method, without explicitly
                        # computing the matrix A.

                        # linear operator: A^H A + lambda_t I
                        def model_normal(m, estimated_mvue=estimated_mvue, lamda_applied=lamda_applied):
                            out = normalize(torch.view_as_real(m).permute(0, -1, 1, 2), estimated_mvue)
                            out = forward_operator(out)
                            out = inverse_operator(out)
                            out = unnormalize(out, estimated_mvue)
                            out = out + lamda_applied.clone() * torch.view_as_real(m).permute(0, -1, 1, 2)
                            out = torch.view_as_complex(out.permute(0, 2, 3, 1))
                            return out

                        cg_solve = ConjugateGradient(model_normal, self.config["num_cg_iter"])

                        meas = forward_operator(samples)  # H x hat t, ref = y
                        zf = inverse_operator(ref)
                        zf = unnormalize(zf, estimated_mvue)
                        zf = zf.type(torch.float32)
                        samples_input = samples
                        samples = samples.to(self.device)

                        # REVERSE DIFFUSION (aka denoise) #
                        # Line 3 in Algo.1
                        samples = samples + step_size * (p_grad) + noise
                        #

                        # Line 4 in Algo.1
                        cg_in = torch.view_as_complex((zf + lamda_applied.clone() * samples).permute(0, 2, 3, 1))
                        samples = cg_solve(torch.view_as_complex(zf.permute(0, 2, 3, 1)), cg_in)
                        #

                        samples = torch.view_as_real(samples).permute(0, -1, 1, 2).type(torch.FloatTensor)
                        if self.config.lambda_func == "learnable":
                            samples = samples.requires_grad_(True)
                        samples = samples.to(self.device)

                        # compute metrics
                        metrics = [step_size, (meas - ref).norm(), (p_grad).abs().mean(), (p_grad).abs().max()]
                        update_pbar_desc(pbar, metrics, pbar_labels)

                        # >>>> Compute the SURE loss
                        # create perturbed input for monte-carlo SURE
                        # Line 5-6 in Algo.1
                        sureloss = SURELoss(
                            perturb_noise=torch.randn_like(samples),
                            eps=torch.abs(zf.max()) / 1000,
                        )

                        # denoise step function with conjugate gradient
                        # the output of this function is the denoised image
                        # This function corresponds to x_{t+1} = h(x_t, \lambda_t) in line 4 of Algo.1 in the paper.
                        # it involves applying the generative model and conjugate gradient
                        # update in sequence. It output a denoised image that confines to the sparse measurement
                        # in the spectral domain.
                        denoise_cg_fn = lambda x: denoise_cg_step(
                            x,
                            score=self.score,
                            labels=labels,
                            step_size=step_size,
                            noise=noise,
                            lambda_t=lamda_applied.clone(),
                            x_zf=zf,
                            cg_solve_fn=cg_solve,
                        )

                        # apply the SURE loss function
                        sure_loss = sureloss(
                            operator=denoise_cg_fn,
                            x=samples_input,
                            y_pseudo_gt=zf,
                            y_ref=samples,
                            complex_input=True,
                        )
                        # --Line 5-6 in Algo.1
                        # <<<< Compute the SURE loss

                        sures.append(sure_loss.detach().cpu().numpy())
                        gt_l2_loss = complex_diff_abs_loss(samples, mvue.squeeze(1))
                        gt_losses.append(gt_l2_loss.detach().cpu().numpy())
                        lambdas.append(lamda.clone().detach().cpu().numpy())

                        init_lambda_update = self.config["init_lambda_update"]
                        last_lambda_update = self.config["last_lambda_update"]
                        if c > init_lambda_update and c < last_lambda_update and self.config.lambda_func == "learnable":
                            # we will use SURE loss to update lambda
                            loss = sure_loss
                            loss.backward(retain_graph=True)
                            optimizer.step()

                        if self.config.lambda_func == "learnable":
                            samples = samples.detach()
                            lamda = lamda.detach()
                            zf = zf.detach()
                            loss = loss.detach()

                    if self.config.early_stop == "stop":
                        # EARLY STOPPING USING SURE LOSS
                        # check the self-validation loss for early stopping
                        if (
                            len(sures) > 3 * window_size
                            and c > 3 * window_size
                            and np.mean(sures[-window_size:]) > np.mean(sures[-2 * window_size : -window_size])
                        ):
                            print("\nAutomatic early stopping activated.")
                            return normalize(samples, estimated_mvue)
                    else:
                        pass

                    if np.isnan((meas - ref).norm().cpu().numpy()):
                        return normalize(samples, estimated_mvue)

                # show images during the generation process to see how the reconstruction evolves
                if (c) % self.config["save_iter"] == 0:
                    estimated_mvue = estimated_mvue.cpu()
                    img_gen = normalize(samples, estimated_mvue)
                    outputs = normalize(samples, estimated_mvue)
                    outputs = outputs.permute(0, 2, 3, 1)
                    outputs = outputs.contiguous()
                    outputs = torch.view_as_complex(outputs)
                    to_display = torch.view_as_complex(
                        img_gen.permute(0, 2, 3, 1)
                        .reshape(-1, self.config["image_size"][0], self.config["image_size"][1], 2)
                        .contiguous()
                    ).abs()
                    if self.config["anatomy"] == "brain":
                        # flip vertically
                        to_display = to_display.flip(-2)
                    elif self.config["anatomy"] == "stanford_knees":
                        # do nothing
                        pass
                    else:
                        pass

                    to_display = scale(to_display)
                    plt.figure()
                    plt.imshow(to_display[0].cpu().numpy(), cmap="gray")
                    plt.title(f'Reconstruction at step {c-self.config["start_iter"]}')
                    plt.show()

        samples = samples.detach()
        return normalize(samples, estimated_mvue)

    def sample(self, y):
        self._initialize()
        mvue = self._sample(y)
        outputs = []
        for i in range(y[0].shape[0]):
            outputs_ = {
                "mvue": mvue[i : i + 1],
            }
            outputs.append(outputs_)
        return outputs
