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

import argparse
import copy
import logging
import os
import json
import random
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses.dice import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.data import DataLoader, CacheDataset, Dataset, decollate_batch
from monai.transforms import (
    AsDiscrete,
    CropForegroundd,
    Compose,
    EnsureType,
    EnsureTyped,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandAdjustContrastd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandFlipd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    SpatialPadd,
)

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Active Learning Setting")

# Directory & Json & Seed
parser.add_argument("--base_dir", default="/home/vishwesh/experiments/al_sanity_test_apr27_2023", type=str)
parser.add_argument("--data_root", default="/scratch_2/data_2021/68111", type=str)
parser.add_argument("--json_path", default="/scratch_2/data_2021/68111/dataset_val_test_0_debug.json", type=str)
parser.add_argument("--seed", default=102, type=int)

# Active learning parameters
parser.add_argument("--active_iters", default=2, type=int)
parser.add_argument("--dropout_ratio", default=0.2, type=float)
parser.add_argument("--mc_number", default=3, type=int)
parser.add_argument("--initial_pool", default=2, type=int)
parser.add_argument("--queries", default=1, type=int)
parser.add_argument("--strategy", default="variance", type=str)  # Options available variance, random

# DL Hyper-parameters
parser.add_argument("--steps", default=30, type=int)
parser.add_argument("--val_interval", default=5, type=int)
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--val_batch_size", default=1, type=int)
parser.add_argument("--lr", default=1e-4, type=float)


def main():
    # Argument parser Code
    # monai.config.print_config()
    args = parser.parse_args()

    print("MAIN Argument values:")
    for k, v in vars(args).items():
        print(k, "=>", v)
        print("-----------------")

    # Set Determinism
    set_determinism(seed=args.seed)

    # Base directory where all model collections will go
    base_model_dir = os.path.normpath(args.base_dir)
    if os.path.exists(args.base_dir) == False:
        os.mkdir(args.base_dir)

    base_model_dir = os.path.join(base_model_dir, "all_models")
    if os.path.exists(base_model_dir) == False:
        os.mkdir(base_model_dir)

    json_base_dir = os.path.join(base_model_dir, "all_jsons")
    if os.path.exists(json_base_dir) == False:
        os.mkdir(json_base_dir)

    fig_base_dir = os.path.join(base_model_dir, "qa_figs")
    if os.path.exists(fig_base_dir) == False:
        os.mkdir(fig_base_dir)

    # Root data path
    data_root = os.path.normpath(args.data_root)

    # Load Json and append root path
    with open(args.json_path, "r") as json_f:
        p_json_data = json.load(json_f)

    p_train_data = p_json_data["training"]
    p_copy_train_data = copy.deepcopy(p_train_data)
    p_val_data = p_json_data["validation"]
    p_copy_val_data = copy.deepcopy(p_val_data)
    p_test_data = p_json_data["test"]
    p_copy_test_data = copy.deepcopy(p_test_data)

    print("Random Strategy is being used for selection instead of SSL based ranking for creating initial pool ...")
    p_grab_indices = random.sample(range(0, len(p_copy_train_data)), args.initial_pool)
    p_labeled_data = []
    p_copy_train_data = np.array(p_copy_train_data)
    p_samples = p_copy_train_data[p_grab_indices]
    for each in p_samples:
        p_labeled_data.append(each)

    p_copy_unl_d = np.delete(p_copy_train_data, p_grab_indices)
    p_copy_unl_d = p_copy_unl_d.tolist()

    print("Updated Json File Sample Count")
    print("Number of Training samples for next iter: {}".format(len(p_labeled_data)))
    print("Number of Unlabeled samples for next iter: {}".format(len(p_copy_unl_d)))

    # Write new json file
    new_json_dict = {}
    new_json_dict["training"] = p_labeled_data
    new_json_dict["unlabeled"] = p_copy_unl_d
    new_json_dict["validation"] = p_copy_val_data
    new_json_dict["test"] = p_copy_test_data

    init_json_f_path = os.path.join(json_base_dir, "json_init_pool_{}.json".format(args.initial_pool))
    with open(init_json_f_path, "w") as j_file:
        json.dump(new_json_dict, j_file)
    j_file.close()

    # Active Json Paths
    new_json_path = ""

    # Previous Ckpt Path
    prev_best_ckpt = ""

    # Model Definition
    device = torch.device("cuda:0")
    network = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
        dropout=args.dropout_ratio,
    )

    network.to(device)

    dice_loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(network.parameters(), args.lr)

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Training & Validation Transforms
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=("image", "label")),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-21,
                a_max=189,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
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
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.50),
            RandAdjustContrastd(keys=["image"], prob=0.50),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.50),
            RandGaussianNoised(keys=["image"], prob=0.50),
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.30),
            RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.30),
            RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.30),
            RandRotate90d(keys=["image", "label"], prob=0.30, max_k=3),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=("image", "label")),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-21,
                a_max=189,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    unl_transforms = Compose(
        [
            LoadImaged(keys=("image")),
            EnsureChannelFirstd(keys=("image")),
            Spacingd(
                keys=("image"),
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear"),
            ),
            ScaleIntensityRanged(keys="image", a_min=-21, a_max=189, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=("image"), source_key="image"),
            EnsureTyped(keys=["image"]),
        ]
    )

    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=3)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=3)])
    # End of Training & Validation Transforms

    # Active Learning iterations
    all_metric_dict = {}
    for active_iter in range(args.active_iters):
        print("Currently on Active Iteration: {}".format(active_iter))

        # Create current active model path
        model_name = "model_" + str(active_iter)
        active_model_dir = os.path.join(base_model_dir, model_name)
        if os.path.exists == False:
            os.mkdir(active_model_dir)

        # Define Summary Writer
        writer = SummaryWriter(log_dir=active_model_dir)

        # Load JSON data
        if active_iter == 0:
            print("Opening json file {} for Active iteration {}".format(init_json_f_path, active_iter))
            with open(init_json_f_path, "rb") as f:
                json_data = json.load(f)
            f.close()

        elif active_iter > 0:
            print("Opening json file {} for Active iteration {}".format(new_json_path, active_iter))
            with open(new_json_path, "rb") as f:
                json_data = json.load(f)
            f.close()

        train_d = json_data["training"]
        copy_train_d = copy.deepcopy(train_d)
        val_d = json_data["validation"]
        copy_val_d = copy.deepcopy(val_d)
        unl_d = json_data["unlabeled"]
        copy_unl_d = copy.deepcopy(unl_d)
        test_d = json_data["test"]
        copy_test_d = copy.deepcopy(test_d)

        # Add data_root to json
        for idx, each_sample in enumerate(train_d):
            train_d[idx]["image"] = os.path.join(data_root, train_d[idx]["image"])
            train_d[idx]["label"] = os.path.join(data_root, train_d[idx]["label"])

        for idx, each_sample in enumerate(val_d):
            val_d[idx]["image"] = os.path.join(data_root, val_d[idx]["image"])
            val_d[idx]["label"] = os.path.join(data_root, val_d[idx]["label"])

        for idx, each_sample in enumerate(unl_d):
            unl_d[idx]["image"] = os.path.join(data_root, unl_d[idx]["image"])
            unl_d[idx]["label"] = os.path.join(data_root, unl_d[idx]["label"])

        for idx, each_sample in enumerate(test_d):
            test_d[idx]["image"] = os.path.join(data_root, test_d[idx]["image"])
            test_d[idx]["label"] = os.path.join(data_root, test_d[idx]["label"])

        train_ds = CacheDataset(data=train_d, transform=train_transforms, cache_rate=1.0)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

        val_ds = CacheDataset(data=val_d, transform=val_transforms, cache_rate=1.0, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=args.val_batch_size)

        test_ds = Dataset(data=test_d, transform=val_transforms)
        test_loader = DataLoader(test_ds, batch_size=args.val_batch_size)

        unl_ds = Dataset(data=unl_d, transform=unl_transforms)
        unl_loader = DataLoader(unl_ds, batch_size=1)

        # Calculation of Epochs based on steps
        max_epochs = np.int(args.steps / (np.ceil(len(train_d) / args.batch_size)))
        print("Epochs Estimated are {} for Active Iter {} with {} Vols".format(max_epochs, active_iter, len(train_d)))

        # Model Training begins for one active iteration
        best_metric = -1
        metric_values = []
        epoch_loss_values = []
        for epoch in range(max_epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            network.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1

                inputs, labels = (
                    batch_data["image"].to(device),
                    batch_data["label"].to(device),
                )

                optimizer.zero_grad()
                outputs = network(inputs)

                loss_dice = dice_loss_function(outputs, labels)

                loss_dice.backward()
                optimizer.step()

                # Total Loss Storage of value
                epoch_loss += loss_dice.item()

                print(f"{step}/{(len(train_ds)) // train_loader.batch_size}, " f"train_loss: {loss_dice.item():.4f}")

            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            writer.add_scalar("Train/Dice Loss", epoch_loss, epoch)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % args.val_interval == 0:
                network.eval()
                with torch.no_grad():
                    for val_data in val_loader:
                        val_inputs, val_labels = (
                            val_data["image"].to(device),
                            val_data["label"].to(device),
                        )
                        roi_size = (160, 160, 160)
                        sw_batch_size = 4
                        val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, network)
                        val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                        val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                        # compute metric for current iteration
                        dice_metric(y_pred=val_outputs, y=val_labels)

                    # aggregate the final mean dice result
                    metric = dice_metric.aggregate().item()
                    # reset the status for next validation round
                    dice_metric.reset()
                    writer.add_scalar("Validation/Dice", metric, epoch)
                    metric_values.append(metric)
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save(network.state_dict(), os.path.join(active_model_dir, "model.pt"))
                        print("saved new best metric model")
                    print(
                        f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                        f"\nbest mean dice: {best_metric:.4f} "
                        f"at epoch: {best_metric_epoch}"
                    )

        # Model training ends for a single active iteration

        print("Loading the final set of trained weights for running inference")
        prev_best_ckpt = os.path.join(active_model_dir, "model.pt")

        device = torch.device("cuda:0")
        ckpt = torch.load(prev_best_ckpt)
        network.load_state_dict(ckpt)
        network.to(device=device)

        # Run metrics on test data for Dice metric
        network.eval()
        with torch.no_grad():
            for test_data in test_loader:
                test_inputs, test_labels = (
                    test_data["image"].to(device),
                    test_data["label"].to(device),
                )
                roi_size = (160, 160, 160)
                sw_batch_size = 4
                test_outputs = sliding_window_inference(test_inputs, roi_size, sw_batch_size, network)
                test_outputs = [post_pred(i) for i in decollate_batch(test_outputs)]
                test_labels = [post_label(i) for i in decollate_batch(test_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=test_outputs, y=test_labels)

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            test_dice = metric
            # reset the status for next validation round
            dice_metric.reset()
            writer.add_scalar("Test/Dice", metric, epoch)

        metric_vals = {"epoch_loss": epoch_loss_values, "validation_dice": metric_values, "test_dice": test_dice}

        # Store all metrics in the overall dict
        all_metric_dict[model_name] = metric_vals
        pickle_f_path = os.path.join(args.base_dir, "all_metrics.pickle")
        with open(pickle_f_path, "wb") as handle:
            pickle.dump(all_metric_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

        # Beginning Acquisition Strategy function
        print("Prepping to run inference on unlabeled pool of data")

        # Please note that the model is being put to 'train' mode explicitly for Monte-Carlo simulations
        network.train()
        print("Weights Loaded and the Network has been put in TRAIN mode, not eval")

        if args.strategy == "variance":
            print("Strategy being used is {}".format(args.strategy))
            scores = {}
            score_list = []
            name_list = []
            print("Running inference for uncertainty ...")
            with torch.no_grad():
                counter = 1
                for unl_data in unl_loader:
                    unl_inputs = unl_data["image"].to(device)

                    roi_size = (160, 160, 160)
                    sw_batch_size = 4

                    accum_unl_outputs = []

                    for mc in range(args.mc_number):
                        unl_outputs = sliding_window_inference(unl_inputs, roi_size, sw_batch_size, network)

                        # Activate the output with Softmax
                        unl_act_outputs = torch.softmax(unl_outputs, dim=1)

                        # Accumulate
                        accum_unl_outputs.append(unl_act_outputs)

                    # Stack it up
                    accum_tensor = torch.stack(accum_unl_outputs)

                    # Squeeze
                    accum_tensor = torch.squeeze(accum_tensor)

                    # Send to CPU
                    accum_numpy = accum_tensor.to("cpu").numpy()
                    accum_numpy = accum_numpy[:, 1:, :, :, :]

                    # The input is assumed with repetitions, channels and then volumetric data
                    vol_input = accum_numpy.astype(dtype="float32")
                    dims = vol_input.shape

                    # Threshold values less than or equal to zero
                    threshold = 0.0005
                    vol_input[vol_input <= 0] = threshold

                    vari = np.nanvar(vol_input, axis=0)
                    variance = np.sum(vari, axis=0)

                    variance = np.expand_dims(variance, axis=0)
                    variance = np.expand_dims(variance, axis=0)

                    variance_dims = np.shape(variance)
                    score_list.append(np.nanmean(variance))
                    name_list.append(unl_data["image_meta_dict"]["filename_or_obj"][0])
                    print(
                        "Variance for image: {} is: {}".format(
                            unl_data["image_meta_dict"]["filename_or_obj"][0], np.nanmean(variance)
                        )
                    )

                    # Plot with matplotlib and save all slices
                    plt.figure(1)
                    plt.imshow(np.squeeze(variance[:, :, np.int(variance_dims[2] / 2)]))
                    plt.colorbar()
                    plt.title("Dropout Uncertainty")
                    fig_path = os.path.join(fig_base_dir, "active_{}_file_{}.png".format(active_iter, counter))
                    plt.savefig(fig_path)
                    plt.clf()
                    plt.close(1)
                    counter = counter + 1

            print("Inference for Uncertainty Complete, working on ranking the unlabeled data")
            # Normalize scores between 0 and 1
            norm_entropy = (score_list - np.min(score_list)) / (np.max(score_list) - np.min(score_list))

            # Put together name and score from the 2 lists
            for ent_score, ent_name in zip(norm_entropy, name_list):
                scores[ent_name] = ent_score

            # Detach values and keys for sorting
            scores_vals = []
            scores_keys = []

            for key, value in scores.items():
                scores_vals_t = value
                score_keys_t = key.split("/")[-2:]
                print("Score Key is {} and value is {}".format(score_keys_t, scores_vals_t))

                score_key_path = os.path.join(score_keys_t[0], score_keys_t[1])

                scores_vals.append(scores_vals_t)
                scores_keys.append(score_key_path)

            sorted_indices = np.argsort(scores_vals)

            # Retrieve most unstable samples list
            most_unstable = sorted_indices[-args.queries :]
            scores_keys = np.array(scores_keys)
            most_unstable_names = scores_keys[most_unstable]
            most_unstable_names = most_unstable_names.tolist()

            rem_indices = sorted_indices[: -args.queries]
            rem_names = scores_keys[rem_indices]
            rem_names = rem_names.tolist()

            copy_unl_d = np.array(copy_unl_d)

            # Form the new JSON
            # Get indices from unlabeled data pool using most unstable names
            grab_indices = []
            for each_unstable_name in most_unstable_names:
                for idx_unl, each_sample in enumerate(copy_unl_d):
                    print(each_sample)
                    if each_unstable_name == each_sample["image"]:
                        grab_indices.append(idx_unl)

            samples = copy_unl_d[grab_indices]
            for each in samples:
                copy_train_d.append(each)

            grab_rem = []
            unlabeled_data = []
            for each_rem_name in rem_names:
                for idx_unl, each_sample in enumerate(copy_unl_d):
                    if each_rem_name == each_sample["image"]:
                        print(each_sample)
                        grab_rem.append(idx_unl)

            rem_samples = copy_unl_d[grab_rem]
            for each in rem_samples:
                unlabeled_data.append(each)
            copy_unl_d = unlabeled_data

            print("Updated Json File Sample Count")
            print("Number of Training samples for next iter: {}".format(len(copy_train_d)))
            print("Number of Unlabeled samples for next iter: {}".format(len(copy_unl_d)))

        elif args.strategy == "random":
            print("Random Strategy is being used for selection")
            grab_indices = random.sample(range(0, len(copy_unl_d)), args.queries)

            copy_unl_d = np.array(copy_unl_d)
            samples = copy_unl_d[grab_indices]
            for each in samples:
                copy_train_d.append(each)

            copy_unl_d = np.delete(copy_unl_d, grab_indices)
            copy_unl_d = copy_unl_d.tolist()

            print("Updated Json File Sample Count")
            print("Number of Training samples for next iter: {}".format(len(copy_train_d)))
            print("Number of Unlabeled samples for next iter: {}".format(len(copy_unl_d)))

        # Write new json file
        new_json_dict = {}
        new_json_dict["training"] = copy_train_d
        new_json_dict["unlabeled"] = copy_unl_d
        new_json_dict["validation"] = copy_val_d
        new_json_dict["test"] = copy_test_d

        new_json_file_path = os.path.join(json_base_dir, "json_iter_{}.json".format(active_iter))
        with open(new_json_file_path, "w") as j_file:
            json.dump(new_json_dict, j_file)
        j_file.close()

        # Update New Json path
        new_json_path = new_json_file_path
        print("Active Iteration {} Completed".format(active_iter))


if __name__ == "__main__":
    main()
