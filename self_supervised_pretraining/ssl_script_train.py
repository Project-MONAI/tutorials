import os
import json
import time
import torch
import matplotlib.pyplot as plt

from torch.nn import L1Loss
from monai.utils import set_determinism, first
from monai.networks.nets import ViT
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    CopyItemsd,
    SpatialPadd,
    EnsureChannelFirstd,
    Spacingd,
    OneOf,
    ScaleIntensityRanged,
    RandSpatialCropSamplesd,
    RandCoarseDropoutd,
    RandCoarseShuffled
)

def main():

    # Defining file paths & output directory path
    json_Path = os.path.normpath('/to/be/defined')
    data_Root = os.path.normpath('/to/be/defined')
    logdir_path = os.path.normpath('/to/be/defined')

    # Load Json & Append Root Path
    with open(json_Path, 'r') as json_f:
        json_Data = json.load(json_f)

    train_Data = json_Data['training']
    val_Data = json_Data['validation']

    for idx, each_d in enumerate(train_Data):
        train_Data[idx]['image'] = os.path.join(data_Root, train_Data[idx]['image'])

    for idx, each_d in enumerate(val_Data):
        val_Data[idx]['image'] = os.path.join(data_Root, val_Data[idx]['image'])

    print('Total Number of Training Data Samples: {}'.format(len(train_Data)))
    print(train_Data)
    print('#' * 10)
    print('Total Number of Validation Data Samples: {}'.format(len(val_Data)))
    print(val_Data)
    print('#' * 10)

    # Set Determinism
    set_determinism(seed=453)

    # Define Training Transforms
    train_Transforms = Compose(
        [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(
            2.0, 2.0, 2.0), mode=("bilinear")),
        ScaleIntensityRanged(
            keys=["image"], a_min=-57, a_max=164,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=(96, 96, 96)),
        RandSpatialCropSamplesd(keys=["image"], roi_size=(96, 96, 96), random_size=False, num_samples=2),
        CopyItemsd(keys=["image"], times=1, names=["gt_image"], allow_missing_keys=False),
        OneOf(transforms=[
            RandCoarseDropoutd(keys=["image"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True,
                               max_spatial_size=32),
            RandCoarseDropoutd(keys=["image"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False,
                               max_spatial_size=64),
            ]
        ),
        RandCoarseShuffled(keys=["image"], prob=0.8, holes=10, spatial_size=8)
        ]
    )

    check_ds = Dataset(data=train_Data, transform=train_Transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    image = (check_data["image"][0][0])
    print(f"image shape: {image.shape}")

    # Define Network ViT backbone & Loss & Optimizer
    device = torch.device("cuda:0")
    model = ViT(
                in_channels=1,
                img_size=(96, 96, 96),
                patch_size=(16, 16, 16),
                pos_embed='conv',
                hidden_size=768,
                mlp_dim=3072,
    )

    model = model.to(device)

    loss_function = L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    # Define DataLoader using MONAI, CacheDataset needs to be used
    train_ds = CacheDataset(data=train_Data, transform=train_Transforms, cache_rate=1.0)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)

    val_ds = CacheDataset(data=val_Data, transform=train_Transforms, cache_rate=1.0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=True, num_workers=4)

    # Define Hyper-paramters for training loop
    max_epochs = 300
    val_interval = 2
    epoch_loss_values = []
    step_loss_values = []
    val_loss_values = []
    best_val_loss = 1000.0

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            start_time = time.time()
            inputs, gt_input = (
                batch_data["image"].to(device),
                batch_data["gt_image"].to(device),
            )
            optimizer.zero_grad()
            outputs, outputs_v2 = model(inputs)
            loss = loss_function(outputs, gt_input)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step_loss_values.append(loss.item())
            end_time = time.time()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, "
                f"train_loss: {loss.item():.4f}, "
                f"time taken: {end_time-start_time}s")

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if epoch % val_interval == 0:
            print('Entering Validation for epoch: {}'.format(epoch+1))
            total_val_loss = 0
            val_step = 0
            model.eval()
            for val_batch in val_loader:
                val_step += 1
                start_time = time.time()
                inputs, gt_input = (
                    val_batch["image"].to(device),
                    val_batch["gt_image"].to(device),
                )
                print('Input shape: {}'.format(inputs.shape))
                outputs, outputs_v2 = model(inputs)
                val_loss = loss_function(outputs, gt_input)
                total_val_loss += val_loss.item()
                end_time = time.time()

            total_val_loss /= step
            val_loss_values.append(total_val_loss)
            print(f"epoch {epoch + 1} Validation average loss: {total_val_loss:.4f}, " f"time taken: {end_time-start_time}s")

            t_dict = {}
            t_dict['step_losses'] = step_loss_values
            t_dict['epoch_losses'] = epoch_loss_values
            t_dict['val_losses'] = val_loss_values

            if total_val_loss < best_val_loss:
                print(f"Saving new model based on validation loss {total_val_loss:.4f}")
                best_val_loss = total_val_loss
                checkpoint = {'epoch': max_epochs,
                              'state_dict': model.state_dict(),
                              'optimizer': optimizer.state_dict()
                              }

                torch.save(checkpoint, os.path.join(logdir_path, 'best_model.pt'))

            plt.figure(1, figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.plot(epoch_loss_values)
            plt.title('Training Loss')
            plt.subplot(1, 2, 2)
            plt.plot(val_loss_values)
            plt.title('Validation Loss')
            plt.savefig(os.path.join(logdir_path, 'loss_plots.png'))
            plt.close(1)

    print('Done')
    return None

if __name__=="__main__":
    main()