# Surgical Tool Localization in endoscopic videos

In this [competition](https://surgtoolloc.grand-challenge.org/Home/), which is a part of MICCAI 2022 in Singapore, participants are challenged to leverage tool presence data as weak labels to train machine learning models to detect and localize tools with bounding boxes in video frames. Team NVIDIA won prizes by finishing [third](https://surgtoolloc.grand-challenge.org/results/) in both categories.

We recreated interesting core components in our solutions while omitting mundane and trivial stuff such as manually removing corrupt videos.



## `preprocess_extract_images_from_video.ipynb`

The raw training data in this competition are 24695 mp4 videos at 60 fps (frames per second), each 30 seconds long. The test data are single frames from vidoes, sampled at 1 fps. There are also other different characteristics between train videos and test images, such as resolution, whether they have empty area on the sides, and whether the bottom menus are blurred.

In this notebook, we extract frame images from training videos and process them to behave the same as test images.


## `preprocess_detect_scene_and_split_fold.ipynb`

The 30 second videos in this dataset are continous segments from surgical procedures. Each surgical operation contain several to dozens of continuous videos. Identifying which videos belong to the same operation (or scene) is important to make fold splits without leakage. We compare image hash of last and first frames to identify scenes, then use iterative stratification to split the folds to make sure each fold contain similar number of videos from similar number of scenes.

