import json
import copy
import math
import os
import numpy as np
import cv2
import torch
from tqdm import tqdm

from skimage.measure import regionprops
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.handlers import (
    CheckpointSaver,
    MeanDice,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    from_engine
)
from monai.inferers import SimpleInferer
from monai.losses import DiceLoss
from monai.networks.nets import BasicUNet
from monai.data import (
    Dataset,
    DataLoader,
)
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsChannelFirstd,
    AsDiscreted,
    Compose,
    EnsureTyped,
    LoadImaged,
    LoadImage,
    RandRotate90d,
    ScaleIntensityRangeD,
    ToNumpyd,
    TorchVisiond,
    ToTensord,
)

from monai.apps.nuclick.transforms import (
    FlattenLabeld,
    ExtractPatchd,
    SplitLabeld,
    AddPointGuidanceSignald,
    FilterImaged
)


def split_pannuke_dataset(image, label, output_dir, groups):
    groups = groups if groups else dict()
    groups = [groups] if isinstance(groups, str) else groups
    if not isinstance(groups, dict):
        groups = {v: k + 1 for k, v in enumerate(groups)}

    label_channels = {
        0: "Neoplastic cells",
        1: "Inflammatory",
        2: "Connective/Soft tissue cells",
        3: "Dead Cells",
        4: "Epithelial",
    }

    print(f"++ Using Groups: {groups}")
    print(f"++ Using Label Channels: {label_channels}")

    images = np.load(image)
    labels = np.load(label)
    print(f"Image Shape: {images.shape}")
    print(f"Labels Shape: {labels.shape}")

    images_dir = output_dir
    labels_dir = os.path.join(output_dir, "labels", "final")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    dataset_json = []
    for i in tqdm(range(images.shape[0])):
        name = f"img_{str(i).zfill(4)}.npy"
        image_file = os.path.join(images_dir, name)
        label_file = os.path.join(labels_dir, name)

        image_np = images[i]
        mask = labels[i]
        label_np = np.zeros(shape=mask.shape[:2])

        for idx, name in label_channels.items():
            if idx < mask.shape[2]:
                m = mask[:, :, idx]
                if np.count_nonzero(m):
                    m[m > 0] = groups.get(name, 1)
                    label_np = np.where(m > 0, m, label_np)

        np.save(image_file, image_np)
        np.save(label_file, label_np)
        dataset_json.append({"image": image_file, "label": label_file})

    return dataset_json

def split_nuclei_dataset(d, centroid_key="centroid", mask_value_key="mask_value", min_area=5):
    dataset_json = []

    mask = LoadImage(image_only=True, dtype=np.uint8)(d["label"])
    _, labels, _, _ = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

    stats = regionprops(labels)
    for stat in stats:
        if stat.area < min_area:
            print(f"++++ Ignored label with smaller area => ( {stat.area} < {min_area})")
            continue

        x, y = stat.centroid
        x = int(math.floor(x))
        y = int(math.floor(y))

        item = copy.deepcopy(d)
        item[centroid_key] = (x, y)
        item[mask_value_key] = stat.label

        dataset_json.append(item)
    return dataset_json

def main():

    # Paths
    img_data_path = os.path.normpath('/scratch/pan_nuke_data/fold_1/Fold_1/images/fold1/images.npy')
    label_data_path = os.path.normpath('/scratch/pan_nuke_data/fold_1/Fold_1/masks/fold1/masks.npy')
    dataset_path = os.path.normpath('/home/vishwesh/nuclick_experiments/try_1/data')
    json_path = os.path.normpath('/home/vishwesh/nuclick_experiments/try_1/data_list.json')
    logging_dir = os.path.normpath('/home/vishwesh/nuclick_experiments/try_6/')
    groups = [
              "Neoplastic cells",
              "Inflammatory",
              "Connective/Soft tissue cells",
              "Dead Cells",
              "Epithelial",
        ]

    #Hyper-params
    patch_size = 128
    min_area = 5

    # Create Dataset
    if os.path.isfile(json_path) == 0:
        dataset_json = split_pannuke_dataset(image=img_data_path,
                                             label=label_data_path,
                                             output_dir=dataset_path,
                                             groups=groups)

        with open(json_path, 'w') as j_file:
            json.dump(dataset_json, j_file)
        j_file.close()
    else:
        with open(json_path, 'r') as j_file:
            dataset_json = json.load(j_file)
        j_file.close()

    ds_json_new = []
    for d in tqdm(dataset_json):
        ds_json_new.extend(split_nuclei_dataset(d, min_area=min_area))

    print('Total DataSize is {}'.format(len(ds_json_new)))
    val_split = round(len(ds_json_new) * 0.8)
    train_ds_json_new = ds_json_new[:val_split]
    val_ds_json_new = ds_json_new[val_split:]

    # Transforms
    train_pre_transforms = Compose(
        [
            LoadImaged(keys=("image", "label"), dtype=np.uint8),
            FilterImaged(keys="image", min_size=5),
            FlattenLabeld(keys="label"),
            AsChannelFirstd(keys="image"),
            AddChanneld(keys="label"),
            ExtractPatchd(keys=("image", "label"), patch_size=patch_size),
            SplitLabeld(label="label", others="others", mask_value="mask_value", min_area=min_area),
            ToTensord(keys="image"),
            TorchVisiond(
                keys="image", name="ColorJitter", brightness=64.0 / 255.0, contrast=0.75, saturation=0.25, hue=0.04
            ),
            ToNumpyd(keys="image"),
            RandRotate90d(keys=("image", "label", "others"), prob=0.5, spatial_axes=(0, 1)),
            ScaleIntensityRangeD(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
            AddPointGuidanceSignald(image="image", label="label", others="others"),
            EnsureTyped(keys=("image", "label"))
        ]
    )

    train_post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold_values=True, logit_thresh=0.5),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=("image", "label"), dtype=np.uint8),
            FilterImaged(keys="image", min_size=5),
            FlattenLabeld(keys="label"),
            AsChannelFirstd(keys="image"),
            AddChanneld(keys="label"),
            ExtractPatchd(keys=("image", "label"), patch_size=patch_size),
            SplitLabeld(label="label", others="others", mask_value="mask_value", min_area=min_area),
            ToTensord(keys="image"),
            TorchVisiond(
                keys="image", name="ColorJitter", brightness=64.0 / 255.0, contrast=0.75, saturation=0.25, hue=0.04
            ),
            ToNumpyd(keys="image"),
            RandRotate90d(keys=("image", "label", "others"), prob=0.5, spatial_axes=(0, 1)),
            ScaleIntensityRangeD(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
            AddPointGuidanceSignald(image="image", label="label", others="others", drop_rate=1.0),
            EnsureTyped(keys=("image", "label"))
        ]
    )

    train_key_metric = {"train_dice": MeanDice(include_background=False, output_transform=from_engine(["pred", "label"]))}
    val_key_metric = {"val_dice": MeanDice(include_background=False, output_transform=from_engine(["pred", "label"]))}
    val_inferer = SimpleInferer()

    # Define Dataset & Loading
    train_data_set = Dataset(train_ds_json_new, transform=train_pre_transforms)
    train_data_loader = DataLoader(
                                   dataset=train_data_set,
                                   batch_size=32,
                                   shuffle=True,
                                   num_workers=2
                                )

    val_data_set = Dataset(val_ds_json_new, transform=val_transforms)
    val_data_loader = DataLoader(
                                   dataset=val_data_set,
                                   batch_size=32,
                                   shuffle=True,
                                   num_workers=2
                                )

    # Network Definition, Optimizer etc
    device = torch.device("cuda")

    network = BasicUNet(
        spatial_dims=2,
        in_channels=5,
        out_channels=1,
        features=(32, 64, 128, 256, 512, 32),
    )

    network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), 0.0001)
    dice_loss = DiceLoss(sigmoid=True, squared_pred=True)

    # Training Process
    val_handlers = [
        # use the logger "train_log" defined at the beginning of this program
        StatsHandler(name="train_log", output_transform=lambda x: None),
        TensorBoardStatsHandler(log_dir=logging_dir, output_transform=lambda x: None),
        TensorBoardImageHandler(
            log_dir=logging_dir,
            batch_transform=from_engine(["image", "label"]),
            output_transform=from_engine(["pred"]),
        ),
        CheckpointSaver(save_dir=logging_dir, save_dict={"network": network}, save_key_metric=True),
    ]

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_data_loader,
        network=network,
        inferer=val_inferer,
        postprocessing=train_post_transforms,
        key_val_metric=val_key_metric,
        val_handlers=val_handlers,
        # if no FP16 support in GPU or PyTorch version < 1.6, will not enable AMP evaluation
        amp=False,
    )

    train_handlers = [
        ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
        # use the logger "train_log" defined at the beginning of this program
        StatsHandler(name="train_log",
                     tag_name="train_loss",
                     output_transform=from_engine(["loss"], first=True)),
        TensorBoardStatsHandler(log_dir=logging_dir,
                                tag_name="train_loss",
                                output_transform=from_engine(["loss"], first=True)
                                ),
        CheckpointSaver(save_dir=logging_dir,
                        save_dict={"net": network, "opt": optimizer},
                        save_interval=1,
                        epoch_level=True),
    ]

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=30,
        train_data_loader=train_data_loader,
        network=network,
        optimizer=optimizer,
        loss_function=dice_loss,
        inferer=SimpleInferer(),
        # if no FP16 support in GPU or PyTorch version < 1.6, will not enable AMP evaluation
        amp=False,
        postprocessing=train_post_transforms,
        key_train_metric=train_key_metric,
        train_handlers=train_handlers,
    )
    trainer.run()

    # End ...
    return None

if __name__=="__main__":
    main()