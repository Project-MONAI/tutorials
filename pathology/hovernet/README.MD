# HoVerNet Examples

This folder contains ignite version examples to run train and validate a HoVerNet model.
It also has torch version notebooks to run training and evaluation.
<p align="center">
  <img src="https://ars.els-cdn.com/content/image/1-s2.0-S1361841519301045-fx1_lrg.jpg" alt="hovernet scheme")
</p>
implementation based on:

Simon Graham et al., HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images.' Medical Image Analysis, (2019). https://arxiv.org/abs/1812.06499

### 1. Data

CoNSeP datasets which are used in the examples can be downloaded from https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/.
- First download CoNSeP dataset to `data_root`.
- Run prepare_patches.py to prepare patches from images.

### 2. Questions and bugs

- For questions relating to the use of MONAI, please us our [Discussions tab](https://github.com/Project-MONAI/MONAI/discussions) on the main repository of MONAI.
- For bugs relating to MONAI functionality, please create an issue on the [main repository](https://github.com/Project-MONAI/MONAI/issues).
- For bugs relating to the running of a tutorial, please create an issue in [this repository](https://github.com/Project-MONAI/Tutorials/issues).


### 3. List of notebooks and examples
#### [Prepare Your Data](./prepare_patches.py)
This example is used to prepare patches from tiles referring to the implementation from https://github.com/vqdang/hover_net/blob/master/extract_patches.py. Prepared patches will be saved in `data_root`/Prepared.

```bash
# Run to know all possible options
python ./prepare_patches.py -h

# Prepare patches from images
python ./prepare_patches.py \
    --root   `data_root`
```

#### [HoVerNet Training](./training.py)
This example uses MONAI workflow to train a HoVerNet model on prepared CoNSeP dataset.
Since HoVerNet is training via a two-stage approach. First initialised the model with pre-trained weights on the [ImageNet dataset](https://ieeexplore.ieee.org/document/5206848), trained only the decoders for the first 50 epochs, and then fine-tuned all layers for another 50 epochs. We need to specify `--stage` during training.

Each user is responsible for checking the content of models/datasets and the applicable licenses and determining if suitable for the intended use.
The license for the pre-trained model used in examples is different than MONAI license. Please check the source where these weights are obtained from:
https://github.com/vqdang/hover_net#data-format


```bash
# Run to know all possible options
python ./training.py -h

# Train a hovernet model on single-gpu(replace with your own ckpt path)
export CUDA_VISIBLE_DEVICES=0; python training.py \
                                --ep 50 \
                                --stage 0 \
                                --bs 16 \
                                --root `save_root`
export CUDA_VISIBLE_DEVICES=0; python training.py \
                                --ep 50 \
                                --stage 1 \
                                --bs 4 \
                                --root `save_root` \
                                --ckpt logs/stage0/checkpoint_epoch=50.pt

# Train a hovernet model on multi-gpu (NVIDIA)(replace with your own ckpt path)
torchrun --nnodes=1 --nproc_per_node=2 training.py \
                                        --ep 50 \
                                        --bs 8 \
                                        --root `save_root` \
                                        --stage 0
torchrun --nnodes=1 --nproc_per_node=2 training.py \
                                        --ep 50 \
                                        --bs 2 \
                                        --root `save_root` \
                                        --stage 1 \
                                        --ckpt logs/stage0/checkpoint_epoch=50.pt
```

#### [HoVerNet Validation](./evaluation.py)
This example uses MONAI workflow to evaluate the trained HoVerNet model on prepared test data from CoNSeP dataset.
With their metrics on original mode. We reproduce the results with Dice: 0.82762; PQ: 0.48976; F1d: 0.73592.
```bash
# Run to know all possible options
python ./evaluation.py -h

# Evaluate a HoVerNet model
python ./evaluation.py
    --root `save_root` \
    --ckpt logs/stage0/checkpoint_epoch=50.pt
```

## Disclaimer

This is an example, not to be used for diagnostic purposes.
