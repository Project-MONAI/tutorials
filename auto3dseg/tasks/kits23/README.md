
# KiTS23 

<img align="right" src="./kits23_example.png" width=30%>

This tutorial shows how to use Auto3DSeg with KiTS 2023 dataset from the MICCAI 2023 challenge **[The 2023 Kidney and Kidney Tumor Segmentation Challenge (KiTS23)](https://kits-challenge.org/kits23/)**. 
The example is based on the 1st place solution in the KiTS challenge [(NVAUTO team)](https://kits-challenge.org/kits23/#kits23-official-results):

Andriy Myronenko, Dong Yang, Yufan He and Daguang Xu: "Automated 3D Segmentation of Kidneys and Tumors in MICCAI KiTS 2023 Challenge". In MICCAI (2023). [arXiv](https://arxiv.org/abs/2310.04110)


## Task overview

The task is to segment kidneys, tumors and cysts from 3D CTs. The dataset contains 489 cases with resolutions rangingn between 0.39x0.39x0.5 and 1x1x5 mm. 
Please download the KiTS23 [dataset](https://kits-challenge.org/kits23/#) and place it in the "/data/kits23" folder to follow this tutorial. 

## Auto3DSeg

With Auto3DSeg most segmentation parameters are automatically determined.  In this tutorial, we start from a basic automated example, then show how different options can be adjusted if necessary.  We use only the  **SegResNet** algo here for simplicity, which is a training recepie based on the [segresnet] (https://docs.monai.io/en/latest/networks.html#segresnetds). 


### Running based on the input config (one-liner)

The Auto3DSeg can be run using a config **input.yaml**

```bash
python -m monai.apps.auto3dseg AutoRunner run --input=./input.yaml --algos=segresnet
```

This one line of code will run the full training workflow, including data analysis, multi-fold training, ensembling. The system will adjust parameters based on the data and your available GPU (or multi-GPU) hardware configuration. 
Here we explicitely specified to use only segresnet algo, for other possible parameters of the AutoRunner please see the [monai docs](https://github.com/Project-MONAI/MONAI/blob/main/monai/apps/auto3dseg/auto_runner.py). 

The [input.yaml](./input.yaml) describes the dataset (KiTS23) task, and must include at least 3 mandatory fields: mondality (CT), dataset location (here it's /data/kits23) and the dataset manifest json file [kits23_folds.json](./kits23_folds.json).
Other parameters can optionally be also added to the input.yaml config.   For KiTS23 dataset specifically, we include the "class_names" key that show the label grouping for the 3 output classes that KiTS23 challenge asks (which is something specific for the KiTS task)

### Running from the code

If you prefer to run the same thing from code (which will allow more customizations), once can create a python file "example.py" and simply run it as 
```bash
python example.py
```
```python
# example.py file

from monai.apps.auto3dseg import AutoRunner

def main():
    runner = AutoRunner(input='./input.yaml', algos = 'segresnet')
    runner.run()

if __name__ == '__main__':
  main()
```

## Validation performance: NVIDIA DGX-1 (8x V100 32G)

The validation results can be obtained by running the training script with MONAI 1.3.0 on NVIDIA DGX-1 with (8x V100 32GB) GPUs. The results below are in terms of average dice.


| | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Avg |
|:------:|:------:|:------:|:------:|:------:|:------:|:---:|
| SegResNet | 0.8997 | 0.8739 | 0.8923 |0.8911 | 0.8892 |0.88924 |


## Data

The KiTS23 challenge dataset [2,3] can be downloaded from [here](https://kits-challenge.org/kits23). Each user is responsible for checking the content of the datasets and the applicable licenses and determining if suitable for the intended use. The license for the KiTS23 dataset is different than the MONAI license.


## References
[1] Andriy Myronenko, Dong Yang, Yufan He and Daguang Xu: "Automated 3D Segmentation of Kidneys and Tumors in MICCAI KiTS 2023 Challenge". In MICCAI (2023). https://arxiv.org/abs/2310.04110


[2] Heller, N., Isensee, F., Maier-Hein, K.H., Hou, X., Xie, C., Li, F., Nan, Y., Mu, G., Lin, Z., Han, M., et al.: The state of the art in kidney and kidney tumor segmentation in contrast-enhanced ct imaging: Results of the kits19 challenge. Medical Image Analysis 67, 101821 (2021)

[3] Heller, N., Wood, A., Isensee, F., Radsch, T., Tejpaul, R., Papanikolopoulos, N.,Weight, C.: The 2023 kidney and kidney tumor segmentation challenge, https://kits-challenge.org/kits23/

