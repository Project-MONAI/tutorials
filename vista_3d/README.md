# MONAI **V**ersatile **I**maging **S**egmen**T**ation and **A**nnotation
[[`Paper`](https://arxiv.org/pdf/2406.05285)] [[`Demo`](https://build.nvidia.com/nvidia/vista-3d)] [[`Checkpoint`](https://developer.download.nvidia.com/assets/Clara/monai/tutorials/model_zoo/model_vista3d.pt)]
## Overview

The **VISTA3D** is a foundation model trained systematically on 11,454 volumes encompassing 127 types of human anatomical structures and various lesions. It provides accurate out-of-the-box segmentation that matches state-of-the-art supervised models which are trained on each dataset. The model also achieves state-of-the-art zero-shot interactive segmentation in 3D, representing a promising step toward developing a versatile medical image foundation model.

The tutorial demonstrates how to finetune the VISTA3D model on user data, where we use the MSD Task09 Spleen as the example.

In summary, the tutorial covers the following:
- Creation of datasets and data transforms for training and validation
- Create a VISTA3D model and load the pretrained checkpoint
- Implementation of the finetuning loop
- Mixed precision training with GradScaler
- Visualization of training loss and validation accuracy
- Inference on a single validation image
- Visualization of input image, ground truth, and model prediction

For more advanced use, please refer to [VISTA3D research codebase](https://github.com/Project-MONAI/VISTA/tree/main/vista3d) and [VISTA3D bundle](https://github.com/Project-MONAI/model-zoo/tree/dev/models/vista3d)
## License

The codebase is under Apache 2.0 Licence. The model weight is released under [NVIDIA OneWay Noncommercial License](./NVIDIA%20OneWay%20Noncommercial%20License.txt).

## Reference

```
@article{he2024vista3d,
  title={VISTA3D: Versatile Imaging SegmenTation and Annotation model for 3D Computed Tomography},
  author={He, Yufan and Guo, Pengfei and Tang, Yucheng and Myronenko, Andriy and Nath, Vishwesh and Xu, Ziyue and Yang, Dong and Zhao, Can and Simon, Benjamin and Belue, Mason and others},
  journal={arXiv preprint arXiv:2406.05285},
  year={2024}
}
```

## Acknowledgement
- [segment-anything](https://github.com/facebookresearch/segment-anything)
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- [VISTA](https://github.com/Project-MONAI/VISTA)
