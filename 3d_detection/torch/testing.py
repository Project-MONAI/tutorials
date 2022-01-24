import argparse
import json

import torch
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
    NormalizeIntensityd
)
from monai.apps.detection.networks.nets.detection.retinanet import retinanet_resnet50_fpn
from monai.apps.detection.networks.nets.detection.anchor_utils import AnchorGenerator


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Testing")
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
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            EnsureTyped(keys=["image", "box", "label"]),
            NormalizeIntensityd(keys=["image"]),
            # Orientationd(keys=["image"], axcodes="RAS"),
            BoxConvertToStandardd(box_keys=["box"],box_mode=gt_box_mode),
            BoxClipToImaged(box_keys=["box"],image_key="image",remove_empty=True),
            EnsureTyped(keys=["image", "box", "label"]),
        ]
    )

    # create a validation data loader
    val_data = load_decathlon_datalist(args.data_list_file_path, is_segmentation = True, data_list_key = "validation", base_dir = args.data_base_dir)
    val_ds = Dataset(
        data = val_data,
        transform = val_transforms,
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available(), collate_fn=no_collation)

    # 3. build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert len(args.returned_layers) == len(args.base_anchor_size)-1
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in args.base_anchor_size)   
    print(anchor_sizes)
    aspect_ratios = [args.base_aspect_ratios] * len(anchor_sizes)
    anchor_generator = AnchorGenerator(args.spatial_dims, anchor_sizes, aspect_ratios)
    model = retinanet_resnet50_fpn(spatial_dims=args.spatial_dims, pretrained=False, progress=True, num_classes=args.num_classes, n_input_channels=args.n_input_channels, pretrained_backbone=False, trainable_backbone_layers=None, anchor_generator=anchor_generator, score_thresh=args.score_thresh, nms_thresh=args.nms_thresh).to(device)
    model.load_state_dict(torch.load(envDict['model_path']))
    
    # 4. apply trained model
    results_dict = {}
    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs = [val_data_i['image'].to(device) for val_data_i in val_data]
            val_outputs = model(val_inputs)
            print(val_outputs)
            exit(0)
            final_outputs = postprocessing(val_outputs)

        results_dict.update(
            {img_id: result for img_id, result in zip(img_ids, final_outputs)}
        )

    # 5. save prediction
    predict_summary = {}
    with open(test_annFile) as test_json:
        # read all test data json
        test_summary = json.load(test_json)
        predict_summary = test_summary
        if labelList is None:
            labelList = test_summary["labelList"]

        if predictMaskRootPath != None:
            predict_summary["root"]["mask"] = predictMaskRootPath
        predict_summary["labelList"] = labelList
        predict_summary["info"]["model"]= model_path
        predict_summary["info"]["model_config"]= model_configFile
        # substitute predicted scores, bbox, mask into predict_summary
        for (subj_id, test_subject_info) in test_summary["subjects"].items():
            prediction = final_predictionsDic[int(subj_id)]
            for (l, label) in enumerate(labelList):
                if l+1 in prediction["score"].keys(): 
                    predict_summary["subjects"][subj_id][label]["pred_confidence"] = prediction["score"][l+1]
                    predict_summary["subjects"][subj_id][label]["pred_bbox"] = prediction["bbox"][l+1]
                    predict_mask_array = prediction["mask"][l+1]
                    if predict_mask_array != None:
                        # *******************************
                        # TO DO: if using mask R-CNN, we need to save the predicted mask array into a nifti file
                        # *******************************
                        predict_mask_file = os.path.join(predict_summary["root"]["mask"],predict_summary["subjects"][subj_id]["dataset"],label,os.path.split(predict_summary["subjects"][subj_id]["data_file"])[-1])
                    else:
                        # if using Faster R-CNN, the mask_file is null
                        predict_summary["subjects"][subj_id][label]["pred_mask_file"] = None
                else:
                    predict_summary["subjects"][subj_id][label]["pred_confidence"] = 0
                    predict_summary["subjects"][subj_id][label]["pred_bbox"] = None
                    predict_summary["subjects"][subj_id][label]["pred_mask_file"] = None
    # save json
    with open(predict_annFile, 'w') as outfile:
        json.dump(predict_summary, outfile, indent = 4)

    # 6. evaluate results
    eveluate_from_json(predict_annFile,test_annFile,maskBool=False)

if __name__ == "__main__":
    main()