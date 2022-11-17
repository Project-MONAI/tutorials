# Surgical Tool Localization in endoscopic videos

In this [competition](https://surgtoolloc.grand-challenge.org/Home/), which is a part of MICCAI 2022 in Singapore, participants are challenged to leverage tool presence data as weak labels to train machine learning models to detect and localize tools with bounding boxes in video frames. Team NVIDIA won prizes by finishing [third](https://surgtoolloc.grand-challenge.org/results/) in both categories.

Within the provided notebooks, we recreated interesting core components in our solutions while omitting mundane and trivial stuff such as manually removing corrupt videos. In order to reproduce our methods, please check the [competition website](https://surgtoolloc.grand-challenge.org/data/) and prepare the dataset first.

Our solution can be divided into three parts: preprocessing, detection and classification.

## Preprocessing

### `preprocess_extract_images_from_video.ipynb`

The raw training data in this competition are 24695 mp4 videos at 60 fps (frames per second), each 30 seconds long. The test data are single frames from vidoes, sampled at 1 fps. There are also other different characteristics between train videos and test images, such as resolution, whether they have empty area on the sides, and whether the bottom menus are blurred.

In this notebook, we extract frame images from training videos and process them to behave the same as test images.

### `preprocess_detect_scene_and_split_fold.ipynb`

The 30 second videos in this dataset are continous segments from surgical procedures. Each surgical operation contain several to dozens of continuous videos. Identifying which videos belong to the same operation (or scene) is important to make fold splits without leakage. We compare image hash of last and first frames to identify scenes, then use iterative stratification to split the folds to make sure each fold contain similar number of videos from similar number of scenes.

### `preprocess_to_build_detection_dataset.ipynb`

Based on the video level labels, we randomly select 1126 frames (covers all 14 tools) and labeled the bounding boxes. The yolo format labels can be downloaded from [here](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/surgtoolloc_tutorial_1126_frame_labels.zip).

In this notebook, we make use of these labels and prepare the yolo format dataset.

## Classification

In order to cast the challenge into a classification task we need to create a dataset with reliable image level labels when only video level labels are given. Because not all frames in a given video will have all tool types from the video level labels, we decided to first use segmentation models to identify frames where we could identify three unique tools per frame. By identifying three tools, we could then apply the video level labels to the given frame. In order to do this we used models published from the MICCAI 2017 Robotic Instrument Segmentation Challenge found [here](https://github.com/ternaus/robot-surgery-segmentation). After applying segmentation models we then used traditional computer vision techniques to count the contours of each unique instrument, and layered additional logic (ie instrument contour needs to touch frame edge) to filter out false positives (The tutorial of this technique will be updated later).

All frames where we could positively identify three unique instruments per frame then formed the basis of our training dataset. The selected frames with cleaned labels is prepared and can be download from [here](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/surgtoolloc_tutorial_cleaned_data.zip).

With the cleaned data, we trained 5 EfficientNet-B4 models (on 5 fold splits) and 5 ConvNext-tiny models, and ensembled them as our final model. In this tutorial, the MONAI EfficientNet-B4 model based training pipeline is prepared, and training methods include cosine learning rate schedule, mixed precision training, mixup augmentations, and weighted loss functions.
Please modify the config file `cfg_efnb4.py` in `classification_files/configs`
according to your actual dataset path. The command to train the model is like:

```
cd classification_files
python train.py --fold 0
```

The cross validation F1 score on this model is around 0.8470.

## Detection

We use [yolov5](https://github.com/ultralytics/yolov5) as our detection model training pipeline. The training script `run_5fold.sh` is in `detection_files`. Please modify the file, especially for `CUDA_VISIBLE_DEVICES` and the `--data` option to based on your actual GPU devices and dataset directory.

The commands to prepare the pipeline is like:
```
git clone https://github.com/ultralytics/yolov5.git
cp detection_files/* yolov5
cd yolov5/
git checkout tags/v6.2
cp data/hyps/hyp.scratch-low.yaml v5m_surg.yaml
```

As for the hyperparameters, we refer to the default `hyp.scratch-low.yaml`, and do the following changes:

1. set initial learning rate `lr0` into `5e-4`.
1. set the rotate degree `degree` into `30.0`.
1. set `shear` into `0.1`.
1. set `mixup` into `0.0`.

After changing the hyperparameters and the script file, you can start trainining via running `bash run_5fold.sh`.
