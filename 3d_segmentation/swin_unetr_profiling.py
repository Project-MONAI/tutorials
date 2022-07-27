# command line: "pip install monai==0.9.1; nsys profile --output /results/orig --force-overwrite true --trace-fork-before-exec true python3 swin_unetr_profiling.py --epochs 5 --val_epochs 5 --batch_size 1 --thread_workers False --num_workers 0"

import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    EnsureTyped,
    ToDeviced
)
from monai.utils import set_determinism

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR

from monai.data import (
#    ThreadDataLoader,
    DataLoader,
    Dataset,
    load_decathlon_datalist,
    decollate_batch,
)

import nvtx
from monai.utils.nvtx import Range

import torch

import argparse

print_config()

parser = argparse.ArgumentParser(description='Profiling Swin UNETR.')
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--val_epochs', default=-1, type=int, help='validation every X epochs; if non-positive value entered, will perform validation only once')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--thread_workers', default=False, type=bool)
parser.add_argument('--num_workers', default=0, type=int)
args = parser.parse_args()
print(args)

assert args.epochs >= 0
assert args.batch_size > 0
assert args.num_workers >= 0

if not args.thread_workers:
    args.num_workers = 0

# +
max_iterations = args.epochs * args.batch_size
max_iterations = max(max_iterations, 24)

eval_num = args.val_epochs * args.batch_size - 1
if eval_num <= 0:
    eval_num = max_iterations - 1
# -



directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)


num_samples = 4
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = Compose(
    [
        Range()(LoadImaged(keys=["image", "label"])),
        Range()(AddChanneld(keys=["image", "label"])),
        Range()(Orientationd(keys=["image", "label"], axcodes="RAS")),
        Range()(Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        )),
        Range()(ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )),
        Range()(CropForegroundd(keys=["image", "label"], source_key="image")),
#         Range()(EnsureTyped(keys=["image", "label"])),
#         Range()(ToDeviced(keys=["image", "label"], device=device)),
        Range()(RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
        )),
        Range()(RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        )),
        Range()(RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        )),
        Range()(RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        )),
        Range()(RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        )),
        Range()(RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        )),
#        Range()(ToTensord(keys=["image", "label"])),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
#        ToTensord(keys=["image", "label"]),
#        EnsureTyped(keys=["image", "label"]),
#        ToDeviced(keys=["image", "label"], device=device),
    ]
)


torch.cuda.empty_cache()
data_dir = "data/"
split_JSON = "dataset_0.json"

datasets = data_dir + split_JSON
datalist = load_decathlon_datalist(datasets, True, "training")
val_files = load_decathlon_datalist(datasets, True, "validation")

# TODO: try thread_workers
train_ds = Dataset(data=datalist, transform=train_transforms)
# train_ds = CacheDataset(data=datalist, transform=train_transforms, cache_num=24, cache_rate=1.0, num_workers=8,)
# train_loader = ThreadDataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
# train_loader = ThreadDataLoader(train_ds, batch_size=args.batch_size, shuffle=True, use_thread_workers=args.thread_workers, num_workers=args.num_workers)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

# val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4)
val_ds = Dataset(data=val_files, transform=val_transforms)
# val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=14,
    feature_size=48,
    use_checkpoint=True,
).to(device)


weight = torch.load("./model_swinvit.pt")
model.load_from(weights=weight)
print("Using pretrained self-supervied Swin UNETR backbone weights !")


torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

def validation(epoch_iterable_val):
    model.eval()
    epoch_iterator_val = iter(epoch_iterable_val)
    with torch.no_grad():
        for _ in range(len(epoch_iterable_val)):
            with nvtx.annotate("val dataload", color="red"):
                batch = next(epoch_iterator_val)
                val_inputs, val_labels = (batch["image"].cuda(device=device), batch["label"].cuda(device=device))

            with nvtx.annotate("sliding window", color="green"):
#                 with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)

            with nvtx.annotate("decollate batch", color="blue"):
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [
                    post_label(val_label_tensor) for val_label_tensor in val_labels_list
                ]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [
                    post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                ]

            with nvtx.annotate("compute metric", color="yellow"):
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)

            epoch_iterable_val.set_description(
                "Validate (%d / %d Steps)" % (global_step, 10.0)
            )

        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterable = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    epoch_iterator = iter(epoch_iterable)

    for step in range(len(epoch_iterable)):
        step += 1

        with nvtx.annotate("dataload", color="red"):
            batch = next(epoch_iterator)
            x, y = (batch["image"].cuda(device=device), batch["label"].cuda(device=device))
#         with torch.cuda.amp.autocast():
        with nvtx.annotate("forward", color="green"):
            logit_map = model(x)
            loss = loss_function(logit_map, y)

        with nvtx.annotate("backward", color="blue"):
            loss.backward()
            epoch_loss += loss.item()

        with nvtx.annotate("update", color="yellow"):
            optimizer.step()
            optimizer.zero_grad()

        epoch_iterable.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)"
            % (global_step, max_iterations, loss)
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterable_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val = validation(epoch_iterable_val)
            # FIXME: epoch_loss is a running average at time of validation??
            with nvtx.annotate("post-validation processing", color="blue"):
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                metric_values.append(dice_val)
                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    torch.save(
                        model.state_dict(), os.path.join(root_dir, "best_metric_model.pth")
                    )
                    print(
                        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
                else:
                    print(
                        "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
        global_step += 1
    return global_step, dice_val_best, global_step_best

# max_iterations = 30000
# eval_num = 500
set_determinism(seed=0)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

begin = time.time()
while global_step < max_iterations:
    with nvtx.annotate("epoch", color="red"):
        global_step, dice_val_best, global_step_best = train(
            global_step, train_loader, dice_val_best, global_step_best
        )
print(f"Total train time: {time.time() - begin:.2f} seconds")


print(
    f"train completed, best_metric: {dice_val_best:.4f} "
    f"at iteration: {global_step_best}"
)

if directory is None:
    shutil.rmtree(root_dir)
