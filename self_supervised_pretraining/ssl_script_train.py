import os
import json
import time
import pickle
import torch
import matplotlib.pyplot as plt

from thop import profile
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
    Orientationd,
    OneOf,
    ScaleIntensityRanged,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    RandCoarseDropoutd,
    RandCoarseShuffled
)

def main():

    # TODO Defining file paths
    #json_Path = os.path.normpath('/scratch/data_2021/tcia_covid19/dataset_split_debug.json')
    json_Path = os.path.normpath('/scratch/data_2021/tcia_covid19/dataset_split.json')
    data_Root = os.path.normpath('/scratch/data_2021/tcia_covid19')
    pickle_path = os.path.normpath('/scratch/data_2021/tcia_covid19/debug_run.pkl')

    # TODO Load Json & Append Root Path
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
    print('#######################')
    print('Total Number of Validation Data Samples: {}'.format(len(val_Data)))
    print(val_Data)
    print('#######################')

    ###############End Of Append Paths#################
    # TODO Set Determinism
    set_determinism(seed=453)

    # TODO Define Training Transforms
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
        #RandSpatialCropd(keys=["image"], roi_size=(96, 96, 96), random_size=False),
        #TODO Ask Wenqi, if I create another copy of the image using CopyItemsd and then pass both 'image_1', 'image_2'
        # through the augmentation transforms, will I get the same exact augmentation or will it be different?
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
    orig_image = (check_data["gt_image"][0][0])
    image = (check_data["image"][0][0])
    print(f"image shape: {image.shape}")

    '''
    # plot the slice [:, :, 80]
    plt.figure("check", (12, 8))
    plt.subplot(2, 3, 1)
    plt.title("Input image")
    plt.imshow(orig_image[:, :, 80], cmap="gray")
    plt.subplot(2, 3, 2)
    plt.title("Input image")
    plt.imshow(orig_image[:, 60, :], cmap="gray")
    plt.subplot(2, 3, 3)
    plt.title("Input image")
    plt.imshow(orig_image[40, :, :], cmap="gray")
    plt.subplot(2, 3, 4)
    plt.title("Aug image")
    plt.imshow(image[:, :, 80], cmap="gray")
    plt.subplot(2, 3, 5)
    plt.title("Aug image")
    plt.imshow(image[:, 60, :], cmap="gray")
    plt.subplot(2, 3, 6)
    plt.title("Aug image")
    plt.imshow(image[40, :, :], cmap="gray")
    plt.show()
    '''

    #TODO Define Network ViT backbone & Loss & Optimizer
    # Also the ViT model will be updated once the ViT_AE PR is merged
    device = torch.device("cuda:0")
    model = ViT(
                in_channels=1,
                img_size=(96, 96, 96),
                patch_size=(16, 16, 16),
                pos_embed='conv',
                hidden_size=768,
                mlp_dim=3072,
                same_as_input_size=True
    )

    #test_input = torch.randn(1, 96, 96, 96)
    #macs, params = profile(model, inputs=(input, ))

    print('Debug here')

    model = model.to(device)

    loss_function = L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    #TODO Define DataLoader using MONAI
    #train_ds = CacheDataset(data=train_Data, transform=train_Transforms, cache_rate=1.0)
    train_ds = Dataset(data=train_Data, transform=train_Transforms)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)

    #val_ds = CacheDataset(data=val_Data, transform=train_Transforms, cache_rate=1.0)
    val_ds = Dataset(data=val_Data, transform=train_Transforms)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=True, num_workers=4)

    #TODO Run the Training Loop
    max_epochs = 300
    val_interval = 2
    best_metric = 1
    best_metric_epoch = 1
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
            with open(pickle_path, 'wb') as f:
                pickle.dump(t_dict, f)
            f.close()

            if total_val_loss < best_val_loss:
                print(f"Saving new model based on validation loss {total_val_loss:.4f}")
                best_val_loss = total_val_loss
                checkpoint = {'epoch': max_epochs,
                              'state_dict': model.state_dict(),
                              'optimizer': optimizer.state_dict()
                              }

                torch.save(checkpoint, '/home/vishwesh/ttl_vit_weights.pt')

            plt.figure(1, figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.plot(epoch_loss_values)
            plt.title('Training Loss')
            plt.subplot(1, 2, 2)
            plt.plot(val_loss_values)
            plt.title('Validation Loss')
            plt.savefig('/home/vishwesh/ssl_pretrain_losses_all_data.png')
            plt.close(1)

    print('Done')

    return None

if __name__=="__main__":
    main()