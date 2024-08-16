# Model Overview
![vista2d.png](vista2d.png)

Vista model training inference pipelines for cell imaging. Min GPU memory requirements for running inference
is 8GB GPU (V100).

Vista-2D was trained on 8x 32GB (V100) GPU, training time: ~24 hours.

Below is a detailed list of all commands for the bundle usage. A quick start guide is located in `Quickstart_guide.md`.

## Bundle Commands
The Vista-2D is based on the template defined in MONAI. For reference, please see the following [documentation](https://docs.monai.io/en/stable/bundle_intro.html).

## Start docker
```
docker pull projectmonai/monai:1.3.1
docker run --gpus all --rm -it --ipc=host --net=host \
  -v <your_bundle_path>:/workspace projectmonai/monai:1.3.1
```

### Install deps
`fastremap 1.14.1` was installed when testing.
- If use docker environment, make sure to install following libraries.
```
pip install fastremap==1.14.1
pip install --no-deps cellpose==3.0.8 natsort==8.4.0 roifile==2024.5.24
pip install git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588#egg=segment_anything
pip install pynvml==11.5.0 #optional for MLFlow support
```
- If use local environment, make sure to install following libraries.
```
pip install torch==2.2.2 monai==1.3.1 numpy==1.24.4
pip install fire==0.6.0 tifffile==2024.5.10 imagecodecs==2024.1.1 pillow==10.2.0 fastremap==1.14.1 numba==0.59.0
pip install --no-deps cellpose==3.0.8 natsort==8.4.0 roifile==2024.5.24
pip install git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588#egg=segment_anything
pip install mlflow==2.13.1 psutil==5.9.4 pynvml==11.5.0 #optional for MLFlow support
```

### Setting
- Ensure dataset format meets requirements in `Quickstart_guide.md`.
Notice that you need to update `config/hyper_parameters.yaml` file with the dataset root path as the basedir and the path to dataset JSON file is required at datalists variable in yaml file.

### Execute training
The best checkpoint after training will be saved in `results/model.pt` by default.
```bash
python -m monai.bundle run_workflow "scripts.workflow.VistaCell" --config_file configs/hyper_parameters.yaml
```

#### Quick run with a few data points
```bash
python -m monai.bundle run_workflow "scripts.workflow.VistaCell" --config_file configs/hyper_parameters.yaml --quick True --train#trainer#max_epochs 3
```

### Execute multi-GPU training
```bash
torchrun --nproc_per_node=gpu -m monai.bundle run_workflow "scripts.workflow.VistaCell" --config_file configs/hyper_parameters.yaml
```

### Execute validation
```bash
python -m monai.bundle run_workflow "scripts.workflow.VistaCell" --config_file configs/hyper_parameters.yaml --pretrained_ckpt_name model.pt --mode eval
```
(can append `--quick True` for quick demoing)

### Execute multi-GPU validation
```bash
torchrun --nproc_per_node=gpu -m monai.bundle run_workflow "scripts.workflow.VistaCell" --config_file configs/hyper_parameters.yaml --mode eval
```

### Execute inference
Inference results will be saved in `results/prediction` directory by default.
- If infer with your own model, use following command.
```bash
python -m monai.bundle run_workflow "scripts.workflow.VistaCell" --config_file configs/hyper_parameters.yaml --mode infer --pretrained_ckpt_name model.pt
```
- If infer with our pretrained model, use following command.
```bash
python -m monai.bundle run_workflow "scripts.workflow.VistaCell" --config_file configs/hyper_parameters.yaml --mode infer --pretrained_ckpt_name vista2d_v1.pt
```
(can append `--quick True` for quick demoing)

### Execute multi-GPU inference
```bash
torchrun --nproc_per_node=gpu -m monai.bundle run_workflow "scripts.workflow.VistaCell" --config_file configs/hyper_parameters.yaml --mode infer --pretrained_ckpt_name model.pt
```
(can append `--quick True` for quick demoing)

### Finetune starting from a trained checkpoint
(we use a smaller learning rate, small number of epochs, and initialize from a checkpoint)
- If finetune your own model, use following command.
```bash
python -m monai.bundle run_workflow "scripts.workflow.VistaCell" --config_file configs/hyper_parameters.yaml --learning_rate=0.001 --train#trainer#max_epochs 20 --pretrained_ckpt_path /path/to/saved/model.pt
```
- If finetune our pretrained model, use following command.
```bash
python -m monai.bundle run_workflow "scripts.workflow.VistaCell" --config_file configs/hyper_parameters.yaml --learning_rate=0.001 --train#trainer#max_epochs 20 --pretrained_ckpt_path /path/to/results/vista2d_v1.pt
```

### Creating JSON for custom datasets
- The Vista-2D bundle uses JSON file lists for training and running inference on models. All JSON datalists that were
used for Vista-2D training can be found in `datalists` directory
- JSON format has mandatory requirements for a `training` list. The list has a dictionary
per data sample that contains 3 keys `image`, `label`, `fold`. `image` contains the relative filename/filepath of the input image, the `label`
contains the instance segmentation filename and `fold` represents if the image will be used for validation or not depending upon which fold is chosen.
- The JSON's use an indent of 4 spaces
- Refer `scripts/generate_json_cellpose.py` for an example on how to create a JSON file
- Please note that once a JSON file is prepped it can be placed in `datalists` directory. The `config/hyper_parameters.yaml` file
will need to be updated with the data root path as the `basedir` and the path to new JSON file is required at `datalists` variable in yaml file.

#### Configuration options

To disable the segmentation writing:
```
--postprocessing []
```

Load a checkpoint for validation or inference (relative path within results directory):
```
--pretrained_ckpt_name "model.pt"
```

Load a checkpoint for validation or inference (absolute path):
```
--pretrained_ckpt_path "/path/to/another/location/model.pt"
```

`--mode eval` or `--mode infer`will use the corresponding configurations from the `validate` or `infer`
of the `configs/hyper_parameters.yaml`.

By default the generated `model.pt` corresponds to the checkpoint at the best validation score,
`model_final.pt` is the checkpoint after the latest training epoch.


### Development

For development you can also run the script directly (same thing)

```bash
python scripts/workflow.py --config_file configs/hyper_parameters.yaml ...
torchrun --nproc_per_node=gpu -m  scripts/workflow.py --config_file configs/hyper_parameters.yaml  ..
```

### MLFlow support

Enable MLFlow logging by specifying "mlflow_tracking_uri" (can be local or remote URL).
If you use local URL, start MLFlow server first.

```bash
mlflow server --host 127.0.0.1 --port 8080
```
You can choose any port that you would like, provided that it’s not already in use.

```bash
python -m monai.bundle run_workflow "scripts.workflow.VistaCell" --config_file configs/hyper_parameters.yaml --mlflow_tracking_uri=http://127.0.0.1:8080
```

Optionally use "--mlflow_run_name=.." to specify MLFlow experiment name, and "--mlflow_log_system_metrics=True/False" to enable logging of CPU/GPU resources (requires pip install psutil pynvml)

Disclaimer
===========
For each dataset, the user is responsible for checking if the dataset license is fit for the intended purpose.

License
=========
The code is released under Apache 2.0.

The model weight is released under CC-BY-NC-SA-4.0.