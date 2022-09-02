<h1 align="center"> Auto3DSeg </h1>

<div align="center"> <img src="figures/workflow.png" width="800"/> </div>

## Introduction

**Auto3DSeg** is a comprehensive solution for large-scale 3D medical image segmentation. It leverages the latest advances in **MONAI** and GPUs to efficiently develop and deploy algorithms with state-of-the-art performance for beginners or advanced researchers in the field. 3D medical image segmentation is an important task with great potential for clinical understanding, disease diagnosis, and surgical planning. According to the statistics of the recent [MICCAI](http://www.miccai.org/) conferences, more than 60% of the papers are applications of segmentation algorithms, and more than half of them use 3D datasets. After working in this field for many years, we have released the state-of-the-art segmentation solution **Auto3DSeg**, which requires minimal user input (e.g., data root and list). The solution offers different levels of user experience for beginners and advanced researchers. It has been tested on large-scale 3D medical imaging datasets in several different modalities.

<details open>
<summary>Major features</summary>

- **Unified Framework**

  **Auto3DSeg** is a self-contained solution for 3D medical image segmentation with minimal user input.

- **Flexible Modular Design**

  **Auto3DSeg** components can be used independently to meet different needs of users.

- **Support of Bring-Your-Own-Algorithm (BYOA)**

  We have introduced an efficient way to introduce users' own algorithms into the **Auto3DSeg** framework.

- **High Accuracy and Efficiency**

  **Auto3DSeg** achieves state-of-the-art performance in most applications of 3D medical image segmentation.

</details>

## What's New

v0.1.0 was released in 9/12/2022:

- Initiaze repository with complete functionalities
- Add four default algorithms for segmentation

Please refer to [changelog.md](docs/changelog.md) for details and release history.

## Get Started

Users needs provide data root and data list as the minimal input.

### Framework Demo

### Component Demo

- Data analyzer
- Algorithm generation
- Unified algorithm API
- Model training, validation, and inference
- Hyper-parameter optimization
- Model ensemble

## Benchmarks

Some benchmark results of public datasets are described in the [tasks](https://github.com/dongyang0122/tutorials/tree/main/auto3dseg/tasks) folder.

## FAQ

Please refer to [FAQ](docs/faq.md) for frequently asked questions.

## Contributing

## Acknowledgement

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@inproceedings{myronenko20183d,
  title={3D MRI brain tumor segmentation using autoencoder regularization},
  author={Myronenko, Andriy},
  booktitle={International MICCAI Brainlesion Workshop},
  pages={311--320},
  year={2018},
  organization={Springer}
}

@inproceedings{he2021dints,
  title={Dints: Differentiable neural network topology search for 3d medical image segmentation},
  author={He, Yufan and Yang, Dong and Roth, Holger and Zhao, Can and Xu, Daguang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5841--5850},
  year={2021}
}

@inproceedings{tang2022self,
  title={Self-supervised pre-training of swin transformers for 3d medical image analysis},
  author={Tang, Yucheng and Yang, Dong and Li, Wenqi and Roth, Holger R and Landman, Bennett and Xu, Daguang and Nath, Vishwesh and Hatamizadeh, Ali},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20730--20740},
  year={2022}
}
```
