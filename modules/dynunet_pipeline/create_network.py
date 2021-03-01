import os

import torch
from monai.networks.nets import DynUNet

from task_params import deep_supr_num, patch_size, spacing


def get_kernels_strides(task_id):
    sizes, spacings = patch_size[task_id], spacing[task_id]
    strides, kernels = [], []

    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides


def get_network(properties, task_id, pretrain_path, checkpoint=None):
    n_class = len(properties["labels"])
    in_channels = len(properties["modality"])
    kernels, strides = get_kernels_strides(task_id)

    net = DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=n_class,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name="instance",
        deep_supervision=True,
        deep_supr_num=deep_supr_num[task_id],
    )

    if checkpoint is not None:
        pretrain_path = os.path.join(pretrain_path, checkpoint)
        if os.path.exists(pretrain_path):
            net.load_state_dict(torch.load(pretrain_path))
            print("pretrained checkpoint: {} loaded".format(pretrain_path))
        else:
            print("no pretrained checkpoint")
    return net
