# Auto3D Tutorial

# Overview

As a monai app, the Auto3D aims to provide an ease-of-use experience for data scientist to automate model training. By using the package, you can automate model selection, hyper parameter tunning, and model ensembling by running a few lines of code. 


# Status
The Auto3D is an improved version of the AutoML app and currently under construction.

Progress:
https://github.com/Project-MONAI/MONAI/issues/4743

# Example Usage

```python
from monai.apps.auto3d.data_analyzer import DataAnalyzer, AutoConfigurer

datalist = {
    "testing": [{"image": "image_003.nii.gz"}],
    "training": [
        {"fold": 0, "image": "image_001.nii.gz", "label": "label_001.nii.gz"},
        {"fold": 0, "image": "image_002.nii.gz", "label": "label_002.nii.gz"},
        {"fold": 1, "image": "image_001.nii.gz", "label": "label_001.nii.gz"},
        {"fold": 1, "image": "image_004.nii.gz", "label": "label_004.nii.gz"},
    ],
}

dataroot = '/datasets' # the directory where you have the image files (in this example we're using nii.gz)
analyser = DataAnalyzer(datalist, dataroot)
datastat = analyser.get_all_case_stats() # it will also generate a data_stats.yaml that saves the stats

input_args = {
    "datastat": datastat,
    "datalist": datalist,
    "dataroot": dataroot,
    "name": "MyTask",
    "task": "segmentation",
    "modality": "MRI",
    "multigpu": True
}

networks = ["UNet", "DiNTS", "SegResNet", "DynuNet"]
configers = []
for net in networks:
    configer = AutoConfiger(net, **input_args)
    configer.generate_scripts()
    configers.append(configer)

trained_models = []
for configer in configers:
    trained_models.append(DistributedTrainers(configer))

for config, model in zipped(configers, trained_models):
    emsembler = ModelEnsembler(config, model)

ensember.infer(datalist["testing"], "save_fig"=True)

```
# Reference:
- Methods(todo, powerpoints)
- 

