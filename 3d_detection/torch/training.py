import argparse
import json
import logging
import sys

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import monai
from monai.config import print_config
from monai.data import Dataset, DataLoader, load_decathlon_datalist
from monai.data.utils import no_collation
from monai.transforms import (
    AddChanneld,
    Compose,
    ScaleIntensityRanged,
    EnsureTyped,
    LoadImaged,
    EnsureTyped,
    BoxConvertToStandardd,
    BoxClipToImaged,
    EnsureChannelFirstd,
    Orientationd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandShiftIntensityd
)

from monai.apps.detection.networks.nets.detection.retinanet import retinanet_resnet50_fpn
from monai.apps.detection.etworks.nets.detection.anchor_utils import AnchorGenerator

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("-e", "--environment-file", default='./config/environment.json', help='environment json file that stores environment path')
    parser.add_argument("-c", "--config-file", default='./config/config_train.json',help='config json file that stores hyper-parameters')
    args = parser.parse_args()
    
    monai.config.print_config()

    envDict = json.load(open(args.environment_file, 'r'))
    configDict = json.load(open(args.config_file, 'r'))

    for k, v in envDict.items():
    	setattr(args, k, v)
    for k, v in configDict.items():
        setattr(args, k, v)

    # 1. define transform
    gt_box_mode="xxyyzz"
    # TO DO: Orientationd for box
    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            EnsureTyped(keys=["image", "box"], dtype=torch.float32),
            EnsureTyped(keys=["label"], dtype=torch.long),
            NormalizeIntensityd(keys=["image"]),
            RandScaleIntensityd(keys=["image"],factors=0.2,prob=0.3),
            RandShiftIntensityd(keys=["image"],offsets=0.2,prob=0.3),
            RandGaussianNoised(keys=["image"],prob=0.3),
            # Orientationd(keys=["image"], axcodes="RAS"),
            BoxConvertToStandardd(box_keys=["box"],box_mode=gt_box_mode),
            BoxClipToImaged(box_keys=["box"],image_key="image",remove_empty=True),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            EnsureTyped(keys=["image", "box"], dtype=torch.float32),
            EnsureTyped(keys=["label"], dtype=torch.long),
            NormalizeIntensityd(keys=["image"]),
            # Orientationd(keys=["image"], axcodes="RAS"),
            BoxConvertToStandardd(box_keys=["box"],box_mode=gt_box_mode),
            BoxClipToImaged(box_keys=["box"],image_key="image",remove_empty=True),
        ]
    )

    
    # 2. prepare training data
    # create a training data loader
    train_data = load_decathlon_datalist(args.data_list_file_path, is_segmentation = True, data_list_key = "training", base_dir = args.data_base_dir)
    train_ds = Dataset(
        data = train_data,
        transform = train_transforms,
    )
    train_loader = DataLoader(train_ds, batch_size=3, shuffle=True, num_workers=1, pin_memory=torch.cuda.is_available(), collate_fn=no_collation)

    # create a validation data loader
    val_data = load_decathlon_datalist(args.data_list_file_path, is_segmentation = True, data_list_key = "validation", base_dir = args.data_base_dir)
    val_ds = Dataset(
        data = val_data,
        transform = val_transforms,
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available(), collate_fn=no_collation)


    # # 3. build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert len(args.returned_layers) == len(args.base_anchor_size)-1
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in args.base_anchor_size)   
    print(anchor_sizes)
    aspect_ratios = [args.base_aspect_ratios] * len(anchor_sizes)
    anchor_generator = AnchorGenerator(args.spatial_dims, anchor_sizes, aspect_ratios)

    model = retinanet_resnet50_fpn(spatial_dims=args.spatial_dims, pretrained=False, progress=True, num_classes=args.num_classes, n_input_channels=args.n_input_channels, pretrained_backbone=False, trainable_backbone_layers=None, anchor_generator=anchor_generator,returned_layers=args.returned_layers, debug=False).to(device)
    # if torch.cuda.is_available():
    #     model = torch.nn.DataParallel(model).cuda()
    # else:
    #     model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    
    # 4. initialize tensorboard writer
    if args.tfevent_path == None:
        tensorboard_writer = SummaryWriter()
    else:
        tensorboard_writer = SummaryWriter(args.tfevent_path)
    
    # 5. train
    val_interval = 2
    best_val_epoch_loss = 1000.0
    best_loss_epoch = -1
    epoch_loss_values = []
    metric_values = []
    max_epochs = 1000
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        # model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs = [batch_data_i['image'].to(device) for batch_data_i in batch_data]
            targets= [dict(labels=batch_data_i['label'].to(device), boxes=batch_data_i['box'].to(device) ) for batch_data_i in batch_data]
            
            optimizer.zero_grad()

            outputs = model(inputs,targets)
            loss = outputs['classification']+outputs['bbox_regression']

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            tensorboard_writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            model.training = True
            val_epoch_loss = 0
            with torch.no_grad():
                num_correct = 0.0
                val_step = 0
                for val_data in val_loader:
                    val_step += 1
                    val_inputs = [val_data_i['image'].to(device) for val_data_i in val_data]
                    val_targets= [dict(labels=val_data_i['label'].to(device), boxes=val_data_i['box'].to(device) ) for val_data_i in val_data]
                    val_outputs = model(val_inputs,targets)
                    val_loss = val_outputs['classification']+val_outputs['bbox_regression']
                    val_epoch_loss += val_loss.item()
                    print(val_outputs)
                
                val_epoch_loss /= (val_step)
                if val_epoch_loss < best_val_epoch_loss:
                    best_val_epoch_loss = val_epoch_loss
                    best_loss_epoch = epoch + 1
                    torch.save(model.state_dict(),
                               envDict['model_path'])
                    print("saved new best metric model")
                print(
                    "current epoch: {} current accuracy: {:.4f} "
                    "best accuracy: {:.4f} at epoch {}".format(
                        epoch + 1, val_epoch_loss, best_val_epoch_loss, best_loss_epoch
                    )
                )
                tensorboard_writer.add_scalar("val_accuracy", val_epoch_loss, epoch + 1)
    print(
        f"train completed, best_metric: {best_val_epoch_loss:.4f} "
        f"at epoch: {best_loss_epoch}")
    tensorboard_writer.close()

if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()