#!/usr/bin/env python

import nibabel as nib
import numpy as np
import os
from copy import deepcopy

from monai.apps.auto3dseg import (
    AlgoEnsembleBestN,
    AlgoEnsembleBuilder,
)
from monai.auto3dseg import Algo
from monai.apps.auto3dseg.ensemble_builder import AlgoEnsembleKeys
from monai.bundle.config_parser import ConfigParser
from monai.transforms import LoadImage

from monai.data import create_test_image_3d

class InferClass:
    def __init__(self):
        pass
    
    def infer(self, image_file):
        # The inference does nothing but just load the image
        filename = image_file['image']
        data = LoadImage(image_only=True)(filename)
        return data

class MyMinimalAlgo(Algo):
    """A minimal demo of how to add your Algo"""
    def __init__(self, output_path: str):
        self.output_path = output_path
        if isinstance(output_path, str) and not os.path.isdir(output_path):
            os.makedirs(output_path)
    
    def train(self):
        # run the neural network training
        best_metric = {"score": float(np.random.rand(1))}
        ConfigParser.export_config_file(
            best_metric, 
            os.path.join(self.output_path, 'progress.yaml'),
            fmt='yaml'
        )
    
    def get_inferer(self):
        infer_class = InferClass
        return infer_class()
    
    def get_score(self):
        dict_file = ConfigParser.load_config_file(os.path.join(self.output_path, "progress.yaml"))
        return dict_file["score"]

    def predict(self, predict_params=None):
        if predict_params is None:
            params = {}
        else:
            params = deepcopy(predict_params)
        files = params.pop("files", ".")
        inferer = self.get_inferer()
        return [inferer.infer(f) for f in files]
    
    def get_output_path(self):
        return self.output_path


# preparation
sim_datalist = {
    "testing": [{"image": "val_001.fake.nii.gz"}, {"image": "val_002.fake.nii.gz"}],
    "training": [
        {"fold": 0, "image": "tr_image_001.fake.nii.gz", "label": "tr_label_001.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_002.fake.nii.gz", "label": "tr_label_002.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_003.fake.nii.gz", "label": "tr_label_003.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_004.fake.nii.gz", "label": "tr_label_004.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_005.fake.nii.gz", "label": "tr_label_005.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_006.fake.nii.gz", "label": "tr_label_006.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_007.fake.nii.gz", "label": "tr_label_007.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_008.fake.nii.gz", "label": "tr_label_008.fake.nii.gz"},
        {"fold": 2, "image": "tr_image_009.fake.nii.gz", "label": "tr_label_009.fake.nii.gz"},
        {"fold": 2, "image": "tr_image_010.fake.nii.gz", "label": "tr_label_010.fake.nii.gz"},
        {"fold": 2, "image": "tr_image_011.fake.nii.gz", "label": "tr_label_011.fake.nii.gz"},
        {"fold": 2, "image": "tr_image_012.fake.nii.gz", "label": "tr_label_012.fake.nii.gz"},
    ],
}

dataroot = os.path.join("./data")
work_dir = os.path.join("./workdir")

da_output_yaml = os.path.join(work_dir, "datastats.yaml")
data_src_cfg = os.path.join(work_dir, "data_src_cfg.yaml")

if not os.path.isdir(dataroot):
    os.makedirs(dataroot)

if not os.path.isdir(work_dir):
    os.makedirs(work_dir)

# Generate a fake dataset
for d in sim_datalist["testing"] + sim_datalist["training"]:
    im, seg = create_test_image_3d(64, 64, 64, rad_max=10, num_seg_classes=1)
    nib_image = nib.Nifti1Image(im, affine=np.eye(4))
    image_fpath = os.path.join(dataroot, d["image"])
    nib.save(nib_image, image_fpath)

    if "label" in d:
        nib_image = nib.Nifti1Image(seg, affine=np.eye(4))
        label_fpath = os.path.join(dataroot, d["label"])
        nib.save(nib_image, label_fpath)

# write to a json file
sim_datalist_filename = os.path.join(dataroot, "sim_datalist.json")
ConfigParser.export_config_file(sim_datalist, sim_datalist_filename)

data_src = {
    "name": "fake_data",
    "task": "segmentation",
    "modality": "MRI",
    "datalist": sim_datalist_filename,
    "dataroot": dataroot,
    "multigpu": False,
    "class_names": ["label_class"],
}

ConfigParser.export_config_file(data_src, data_src_cfg)

## algorithm generation
history = [
    {'my_algo_0': MyMinimalAlgo(os.path.join(work_dir, 'my_algo_0'))},
    {'my_algo_1': MyMinimalAlgo(os.path.join(work_dir, 'my_algo_1'))},
]

## model training
for i, record in enumerate(history):
    for name, algo in record.items():
        algo.train()

## model ensemble
n_best = 1
builder = AlgoEnsembleBuilder(history, data_src_cfg)
builder.set_ensemble_method(AlgoEnsembleBestN(n_best=1))
ensemble = builder.get_ensemble()
pred = ensemble()
print("ensemble picked the following best {0:d}:".format(n_best))
for algo in ensemble.get_algo_ensemble():
    print(algo[AlgoEnsembleKeys.ID])
