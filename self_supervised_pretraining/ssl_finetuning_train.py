import os
import shutil
import tempfile
import argparse
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
)

from monai.config import print_config
from monai.metrics import DiceMetric, compute_meandice
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    Dataset,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)


import torch

def main():
    print_config()

    def validation(epoch_iterator_val):
        model.eval()
        dice_vals = list()
        dice_0 = []
        dice_1 = []
        dice_2 = []
        dice_3 = []
        dice_4 = []
        dice_5 = []
        dice_6 = []
        dice_7 = []
        dice_8 = []
        dice_9 = []
        dice_10 = []
        dice_11 = []
        dice_12 = []
        dice_no_bg = []

        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator_val):
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [
                    post_label(val_label_tensor) for val_label_tensor in val_labels_list
                ]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [
                    post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                ]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                dice = dice_metric.aggregate().item()
                dice_vals.append(dice)
                epoch_iterator_val.set_description(
                    "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
                )

                val_output_convert_1 = val_output_convert[0].unsqueeze(0)
                val_labels_convert_1 = val_labels_convert[0].unsqueeze(0)
                all_organ_dice = compute_meandice(val_output_convert_1, val_labels_convert_1, include_background=False)
                all_organ_dice = np.squeeze(all_organ_dice.detach().cpu().numpy())

                dice_0.append(all_organ_dice[0])
                dice_1.append(all_organ_dice[1])
                dice_2.append(all_organ_dice[2])
                dice_3.append(all_organ_dice[3])
                dice_4.append(all_organ_dice[4])
                dice_5.append(all_organ_dice[5])
                dice_6.append(all_organ_dice[6])
                dice_7.append(all_organ_dice[7])
                dice_8.append(all_organ_dice[8])
                dice_9.append(all_organ_dice[9])
                dice_10.append(all_organ_dice[10])
                dice_11.append(all_organ_dice[11])
                dice_12.append(all_organ_dice[12])
                dice_no_bg.append(np.nanmean(all_organ_dice))

            dice_metric.reset()

            per_organ_dice_mean = [
                np.nanmean(dice_0), np.nanmean(dice_1), np.nanmean(dice_2),
                np.nanmean(dice_3), np.nanmean(dice_4), np.nanmean(dice_5),
                np.nanmean(dice_6), np.nanmean(dice_7), np.nanmean(dice_8),
                np.nanmean(dice_9), np.nanmean(dice_10), np.nanmean(dice_11),
                np.nanmean(dice_12)
            ]
            dice_mean_no_bg = np.nanmean(dice_no_bg)

        mean_dice_val = np.mean(dice_vals)
        return mean_dice_val, dice_mean_no_bg, per_organ_dice_mean

    def train(global_step, train_loader, dice_val_best, global_step_best):
        model.train()
        epoch_loss = 0
        step = 0
        epoch_iterator = tqdm(
            train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )
        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = (batch["image"].cuda(), batch["label"].cuda())
            logit_map = model(x)
            loss = loss_function(logit_map, y)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
            )
            writer.add_scalar("train/loss", scalar_value=loss, global_step=global_step)

            if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
                epoch_iterator_val = tqdm(
                    val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
                )
                dice_val,  dice_no_bg, dice_per_organ = validation(epoch_iterator_val)

                writer.add_scalar("Validation/Mean Dice BTCV", scalar_value=dice_val, global_step=global_step)
                writer.add_scalar("train/loss", scalar_value=loss, global_step=global_step)

                writer.add_scalar("Validation/Mean Dice BTCV No Bg", scalar_value=dice_no_bg, global_step=global_step)
                writer.add_scalar("Validation/Mean Organ 0 Dice", scalar_value=dice_per_organ[0], global_step=global_step)
                writer.add_scalar("Validation/Mean Organ 1 Dice", scalar_value=dice_per_organ[1], global_step=global_step)
                writer.add_scalar("Validation/Mean Organ 2 Dice", scalar_value=dice_per_organ[2], global_step=global_step)
                writer.add_scalar("Validation/Mean Organ 3 Dice", scalar_value=dice_per_organ[3], global_step=global_step)
                writer.add_scalar("Validation/Mean Organ 4 Dice", scalar_value=dice_per_organ[4], global_step=global_step)
                writer.add_scalar("Validation/Mean Organ 5 Dice", scalar_value=dice_per_organ[5], global_step=global_step)
                writer.add_scalar("Validation/Mean Organ 6 Dice", scalar_value=dice_per_organ[6], global_step=global_step)
                writer.add_scalar("Validation/Mean Organ 7 Dice", scalar_value=dice_per_organ[7], global_step=global_step)
                writer.add_scalar("Validation/Mean Organ 8 Dice", scalar_value=dice_per_organ[8], global_step=global_step)
                writer.add_scalar("Validation/Mean Organ 9 Dice", scalar_value=dice_per_organ[9], global_step=global_step)
                writer.add_scalar("Validation/Mean Organ 10 Dice", scalar_value=dice_per_organ[10], global_step=global_step)
                writer.add_scalar("Validation/Mean Organ 11 Dice", scalar_value=dice_per_organ[11], global_step=global_step)
                writer.add_scalar("Validation/Mean Organ 12 Dice", scalar_value=dice_per_organ[12], global_step=global_step)

                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                metric_values.append(dice_val)
                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    torch.save(
                        model.state_dict(), os.path.join(logdir, "best_metric_model.pth")
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

                plt.figure(1, (12, 6))
                plt.subplot(1, 2, 1)
                plt.title("Iteration Average Loss")
                x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
                y = epoch_loss_values
                plt.xlabel("Iteration")
                plt.plot(x, y)
                plt.grid()
                plt.subplot(1, 2, 2)
                plt.title("Val Mean Dice")
                x = [eval_num * (i + 1) for i in range(len(metric_values))]
                y = metric_values
                plt.xlabel("Iteration")
                plt.plot(x, y)
                plt.grid()
                plt.savefig(os.path.join(logdir, 'btcv_finetune_quick_update.png'))
                plt.clf()
                plt.close(1)

            global_step += 1
        return global_step, dice_val_best, global_step_best

    parser = argparse.ArgumentParser(description='UNETR Training')
    parser.add_argument('--logdir', default='/to/be/defined', type=str)
    parser.add_argument('--num_steps', default=30000, type=int)
    parser.add_argument('--eval_num', default=100, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--sw_batch_size', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--use_pretrained', type=int, default=1)
    parser.add_argument('--pretrained_path', type=str, default='/this/has/to/be/pretrained/ssl/weights')
    parser.add_argument('--json_path', default='/use/one/provided/json/file', type=str)
    parser.add_argument('--data_root', default='/to/be/defined', type=str)

    args = parser.parse_args()
    logdir = args.logdir
    if os.path.exists(logdir)==False:
        os.mkdir(logdir)

    # TODO Here are all the transforms
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )

    data_dir = args.data_root
    #split_JSON = "dataset_0.json"
    #datasets = os.path.join(data_dir, split_JSON)
    datasets = args.json_path
    datalist = load_decathlon_datalist(base_dir=data_dir,data_list_file_path=datasets, is_segmentation=True, data_list_key="training")
    val_files = load_decathlon_datalist(base_dir=data_dir, data_list_file_path=datasets, is_segmentation=True, data_list_key="validation")
    train_ds = CacheDataset(
        data=datalist,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=4,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_num=6,
        cache_rate=1.0,
        num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    case_num = 0
    img = val_ds[case_num]["image"]
    label = val_ds[case_num]["label"]
    img_shape = img.shape
    label_shape = label.shape
    print(f"image shape: {img_shape}, label shape: {label_shape}")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNETR(
        in_channels=1,
        out_channels=14,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="conv",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    )

    # TODO Weight Load of ViT here
    if args.use_pretrained==1:
        print('Loading Weights from the Path {}'.format(args.pretrained_path))
        vit_weight_path = args.pretrained_path
        vit_dict = torch.load(vit_weight_path)
        vit_weights = vit_dict['state_dict']

        # TODO Delete the following variable names conv3d_transpose.weight, conv3d_transpose.bias, conv3d_transpose_1.weight, conv3d_transpose_1.bias
        vit_weights.pop('conv3d_transpose_1.bias')
        vit_weights.pop('conv3d_transpose_1.weight')
        vit_weights.pop('conv3d_transpose.bias')
        vit_weights.pop('conv3d_transpose.weight')

        model.vit.load_state_dict(vit_weights)
        print('Pretrained Weights Succesfully Loaded !')

    elif args.use_pretrained==0:
        print('No weights were loaded, all weights being used are from scratch')

    model.to(device)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    writer = SummaryWriter(logdir=logdir)
    max_iterations = args.num_steps
    eval_num = args.eval_num
    post_label = AsDiscrete(to_onehot=True, num_classes=14)
    post_pred = AsDiscrete(argmax=True, to_onehot=True, num_classes=14)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(
            global_step, train_loader, dice_val_best, global_step_best
        )
    model.load_state_dict(torch.load(os.path.join(logdir, "best_metric_model.pth")))

    print(
        f"train completed, best_metric: {dice_val_best:.4f} "
        f"at iteration: {global_step_best}"
    )

    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Iteration Average Loss")
    x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [eval_num * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.grid()
    plt.savefig(os.path.join(logdir, 'btcv_finetune.png'))


if __name__=="__main__":
    main()