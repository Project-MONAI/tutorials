## Quick Start Guide

### Running Inference

#### 1.) Install all dependencies

###### start docker
```
docker pull projectmonai/monai:1.3.1
docker run --gpus all --rm -it --ipc=host --net=host \
  -v <your_bundle_path>:/workspace projectmonai/monai:1.3.1
```

###### Library Dependencies
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

#### 2.) Obtain a public dataset Or Create a custom dataset JSON

###### Retrieve Cellpose public dataset

Trying on your or a new dataset is encouraged. If you do not have access to data, the [Cellpose dataset](https://www.cellpose.org/dataset)
is fairly small and quick to download. Below are instructions on how to download the data

![cellpose_links.png](cellpose_links.png)
Click on train.zip and test.zip to download both directories independently. They both need to be
placed in a `cellpose_dataset` directory. The `cellpose_dataset` will have to be created by the user
in the root data directory. For e.g we can use `/user/cellpose_dataset`.

Ensure that both `train.zip` & `test.zip` have been extracted from the zip format. Below is an example screenshot
of how cellpose dataset directory should look like.

![cellpose_dir.png](cellpose_dir.png)

##### Creating JSON for custom datasets
- The Vista-2D bundle uses JSON file lists for training and running inference on models. All JSON datalists that were
used for Vista-2D training can be found in `datalists` directory
- JSON format has mandatory requirements for a `training` list. The list has a dictionary
per data sample that contains 3 keys `image`, `label`, `fold`. `image` contains the relative filename/filepath of the input image, the `label`
contains the instance segmentation filename and `fold` represents if the image will be used for validation or not depending upon which fold is chosen.
- The JSON's use an indent of 4 spaces
- Refer `scripts/generate_json_cellpose.py` for an example on how to create a JSON file
- Please note that once a JSON file is prepped it can be placed in `datalists` directory. The `config/hyper_parameters.yaml` file
will need to be updated with the data root path as the `basedir` and the path to new JSON file is required at `datalists` variable in yaml file.

#### 3.) Set Paths for data & execute inference
- Set the `basedir` variable to the dataset path in the `cell_vista_segmentation/configs/quick_hyper_params.yaml` file.
- If infer with your own model, use following command.
 ```bash
  python -m monai.bundle run_workflow "scripts.workflow.VistaCell" --config_file configs/quick_hyper_params.yaml --mode infer --pretrained_ckpt_name model.pt
  ```
- If infer with our pretrained model, use following command.
```bash
python -m monai.bundle run_workflow "scripts.workflow.VistaCell" --config_file configs/hyper_parameters.yaml --mode infer --pretrained_ckpt_name vista2d_v1.pt
```
- Ensure that you are setting up the correct model path for the variable `pretrained_ckpt_name`
- The predictions after running the inference are stored in the `results` directory. Below is an example structure of how they are stored

![result_pred_log.png](result_pred_log.png)

### Fine-tuning Vista-2D

#### 1.) Follow Step 1 and 2 from the `Running Inference` section above.

#### 2.) Set Path for data and execute fine-tuning training
-   Ensure that the correct model path is set for the `pretrained_ckpt_path`
- ```bash
  python -m monai.bundle run_workflow "scripts.workflow.VistaCell" --config_file configs/quick_hyper_params.yaml --learning_rate=0.001 --train#trainer#max_epochs 20 --pretrained_ckpt_path /path/to/saved/model.pt
  ```
- The logs and resulting model weights from fine-tuning are stored in `results` directory of the bundle (see screenshot above)
- If you want to finetune our pretrained model
```bash
  python -m monai.bundle run_workflow "scripts.workflow.VistaCell" --config_file configs/quick_hyper_params.yaml --learning_rate=0.001 --train#trainer#max_epochs 20 --pretrained_ckpt_path /path/to/saved/vista2d_v1.pt
  ```
NOTE: The Vista-2D model is trained with ~15k images from different sources, this section uses cellpose data to show an
example on how to finetune Vista-2D. Using cellpose data only will not reproduce the Vista-2D pre-trained weights.

Disclaimer
===========
For each dataset, the user is responsible for checking if the dataset license is fit for the intended purpose.

License
=========
The code is released under Apache 2.0.

The model weight is released under CC-BY-NC-SA-4.0.