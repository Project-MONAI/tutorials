import os
import json
import time
import torch
import matplotlib.pyplot as plt

from torch.nn import L1Loss
from monai.utils import set_determinism, first
from monai.networks.nets import ViTAutoEnc
from monai.losses import ContrastiveLoss
from monai.data import DataLoader, Dataset
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

    #TODO Defining file paths & output directory path
    json_Path = os.path.normpath('/to/be/defined')
    data_Root = os.path.normpath('/to/be/defined')
    logdir_path = os.path.normpath('/to/be/defined')

    if os.path.exists(logdir_path)==False:
        os.mkdir(logdir_path)

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
        CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),
        OneOf(transforms=[
            RandCoarseDropoutd(keys=["image", "image_2"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True,
                               max_spatial_size=32),
            RandCoarseDropoutd(keys=["image", "image_2"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False,
                               max_spatial_size=64),
            ]
        ),
        RandCoarseShuffled(keys=["image", "image_2"], prob=0.8, holes=10, spatial_size=8)
        ]
    )

    check_ds = Dataset(data=train_Data, transform=train_Transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    image = (check_data["image"][0][0])
    print(f"image shape: {image.shape}")

    # Define Network ViT backbone & Loss & Optimizer
    device = torch.device("cuda:0")
    model = ViTAutoEnc(
                in_channels=1,
                img_size=(96, 96, 96),
                patch_size=(16, 16, 16),
                pos_embed='conv',
                hidden_size=768,
                mlp_dim=3072,
    )

    model = model.to(device)

    # Define Hyper-paramters for training loop
    max_epochs = 500
    val_interval = 2
    batch_size = 4
    lr = 1e-4
    epoch_loss_values = []
    step_loss_values = []
    epoch_cl_loss_values = []
    epoch_recon_loss_values = []
    val_loss_values = []
    best_val_loss = 1000.0

    recon_loss = L1Loss()
    contrastive_loss = ContrastiveLoss(batch_size=batch_size*2, temperature=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Define DataLoader using MONAI, CacheDataset needs to be used
    train_ds = Dataset(data=train_Data, transform=train_Transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    val_ds = Dataset(data=val_Data, transform=train_Transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        epoch_cl_loss = 0
        epoch_recon_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            start_time = time.time()

            inputs, inputs_2, gt_input = (
                batch_data["image"].to(device),
                batch_data["image_2"].to(device),
                batch_data["gt_image"].to(device),
            )
            optimizer.zero_grad()
            outputs_v1, hidden_v1 = model(inputs)
            outputs_v2, hidden_v2 = model(inputs_2)

            flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=4)
            flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=4)

            r_loss = recon_loss(outputs_v1, gt_input)
            cl_loss = contrastive_loss(flat_out_v1, flat_out_v2)

            # Adjust the CL loss by Recon Loss
            total_loss = r_loss + cl_loss * r_loss

            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            step_loss_values.append(total_loss.item())

            # CL & Recon Loss Storage of Value
            epoch_cl_loss += cl_loss.item()
            epoch_recon_loss += r_loss.item()

            end_time = time.time()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, "
                f"train_loss: {total_loss.item():.4f}, "
                f"time taken: {end_time-start_time}s")

        epoch_loss /= step
        epoch_cl_loss /= step
        epoch_recon_loss /= step

        epoch_loss_values.append(epoch_loss)
        epoch_cl_loss_values.append(epoch_cl_loss)
        epoch_recon_loss_values.append(epoch_recon_loss)
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
                val_loss = recon_loss(outputs, gt_input)
                total_val_loss += val_loss.item()
                end_time = time.time()

            total_val_loss /= step
            val_loss_values.append(total_val_loss)
            print(f"epoch {epoch + 1} Validation average loss: {total_val_loss:.4f}, " f"time taken: {end_time-start_time}s")

            if total_val_loss < best_val_loss:
                print(f"Saving new model based on validation loss {total_val_loss:.4f}")
                best_val_loss = total_val_loss
                checkpoint = {'epoch': max_epochs,
                              'state_dict': model.state_dict(),
                              'optimizer': optimizer.state_dict()
                              }
                torch.save(checkpoint, os.path.join(logdir_path, 'best_model.pt'))

            plt.figure(1, figsize=(8, 8))
            plt.subplot(2, 2, 1)
            plt.plot(epoch_loss_values)
            plt.grid()
            plt.title('Training Loss')

            plt.subplot(2, 2, 2)
            plt.plot(val_loss_values)
            plt.grid()
            plt.title('Validation Loss')

            plt.subplot(2, 2, 3)
            plt.plot(epoch_cl_loss_values)
            plt.grid()
            plt.title('Training Contrastive Loss')

            plt.subplot(2, 2, 4)
            plt.plot(epoch_recon_loss_values)
            plt.grid()
            plt.title('Training Recon Loss')

            plt.savefig(os.path.join(logdir_path, 'loss_plots.png'))
            plt.close(1)

    print('Done')
    return None

if __name__=="__main__":
    main()
