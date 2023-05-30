# Easy Integrate Bundle Tutorial

A MONAI bundle contains the stored weights of a model, training, inference, post-processing transform sequences and other information. This tutorial aims to demonstrate users how to quickly integrate the bundle into their own application.
The tutorial create a straightforward ensemble application and instruct users on how to use the existing bundle.

The example training dataset is Task09_Spleen.tar from http://medicaldecathlon.com/.

## Requirements

The script is tested with:

- `Ubuntu 20.04` | `Python 3.8.13` | `CUDA 11.7` | `Pytorch 1.11.0`

- the default pipeline requires at least 8GB memory per gpu

- it is tested on 24gb single-gpu machine

## Dependencies and installation

### MONAI

You can conda environments to install the dependencies.

```bash
pip install scikit-learn==0.24.2
```

or you can just use MONAI docker.
```bash
docker pull projectmonai/monai:latest
```

For more information please check out [the installation guide](https://docs.monai.io/en/latest/installation.html).

## Examples

Check all possible options

```bash
python ./ensemble.py -h
```
### Get started
1. Prepare your bundle.

    First download a bundle to somewhere as your `bundle_root_path`:

    ```shell
    python -m monai.bundle download --name spleen_ct_segmentation --bundle_dir "./"
    ```

2. Prepare your data.

    Put your data in `data_root_path`. We prefer you organize your dataset as MSD datasets structure. Then, split your dataset into train and test subsets, and generate a json file named `dataset.json` under the `data_root_path`, like:
    ```
     {
        "training": [
            {
                "image": "./image1.nii.gz"
                "label": "./label1.nii.gz"
            },
            {
                "image": "./image2.nii.gz",
                "label": "./label2.nii.gz"
            },
            ...
        ],
        "test": [
            {
                "image": "./image.nii.gz"
            },
            ...
        ]
    }
    ```
    The data in training will random split into `n_splits` which you can specify with `--n_splits xx`

3. Run the script. Make sure `bundle_root_path` and `data_root_path` is correct.

```bash
python ensemble.py --bundle_root bundle_root_path --dataset_dir data_root_path
--ensemble Mean
```

## **How to integrate Bundle in your own application**
### Get component from bundle
```
from monai.bundle import ConfigWorkflow

train_workflow = ConfigWorkflow(
    config_file=bundle_config_path,
    meta_file=bundle_metadata_path,
    logging_file=bundle_logging_path,
    workflow="train",
)
train_workflow.initialize()
postprocessing = train_workflow.train_postprocessing
```
### Use component in your pipeline
```
# Notice that the `postprocessing` got from `train_workflow` is instantiated.

evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=your_dataloader,
            network=your_networks,
            inferer=SimpleInferer(),
            postprocessing=postprocessing,
        )
```
### Update component with your own args
```
# update `max_epochs` in workflow
train_workflow.max_epochs = max_epochs
```

## Questions and bugs

- For questions relating to the use of MONAI, please use our [Discussions tab](https://github.com/Project-MONAI/MONAI/discussions) on the main repository of MONAI.
- For bugs relating to MONAI functionality, please create an issue on the [main repository](https://github.com/Project-MONAI/MONAI/issues).
- For bugs relating to the running of a tutorial, please create an issue in [this repository](https://github.com/Project-MONAI/Tutorials/issues).
