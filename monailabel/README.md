# MONAI Label End-To-End Tutorial Series

This folder contains end-to-end tutorials of MONAI Label applications. Sample apps including `radiology`, `pathology`, `endoscopy` and `monaibundle`.

![image](https://raw.githubusercontent.com/Project-MONAI/MONAILabel/main/docs/images/sampleApps_index.jpeg)

These tutorials demonstrate step-by-step installations and operations of MONAI Label server, client viewers, or datastore platforms.

These notebooks can be downloaded and used for demonstration on local machines. Each one includes detailed installation steps for viewers, datastore, and MONAI Label server setup.

Each tutorial is associated with a sample dataset for non-commercial use. APIs are provided for users to download datasets directly.

Each tutorial provides MONAI Label plugin installation steps associated with the viewer.

## Content

- **Radiology App**:
  - Viewer: [3D Slicer](https://www.slicer.org/) | Datastore: Local | Task: Segmentation
    - [MONAILabel: HelloWorld](monailabel_HelloWorld_radiology_3dslicer.ipynb): Spleen segmentation with 3D Slicer setups.
- **MONAIBUNDLE App**:
  - Viewer: [3D Slicer](https://www.slicer.org/) | Datastore: Local | Task: Segmentation
    - [MONAILabel: Pancreas Tumor Segmentation with 3D Slicer](monailabel_pancreas_tumor_segmentation_3DSlicer.ipynb): Pancreas and tumor segmentation with CT scans in 3D Slicer.
    - [MONAILabel: Multi-organ Segmentation with 3D Slicer](monailabel_monaibundle_3dslicer_multiorgan_seg.ipynb): Multi-organ segmentation with CT scans in 3D Slicer.
- **Pathology App**:
  - Viewer: [QuPath](https://qupath.github.io/) | Datastore: Local | Task: Segmentation
    - [MONAILabel: Nuclei Segmentation with QuPath](monailabel_pathology_nuclei_segmentation_QuPath.ipynb) Nuclei segmentation with QuPath setup and Nuclick models.
- **Endoscopy App**:
  - Viewer: [CVAT](https://github.com/opencv/cvat) | Datastore: Local | Task: Segmentation
    - [MONAILabel: Tooltracking with CVAT](monailabel_endoscopy_cvat_tooltracking.ipynb): Surgical tool segmentation with CVAT/Nuclio setup.


## Hardware Requirements

MONAI Label is supported on most NVIDIA GPUs such as RTX series, A100, H100, A30, A10, V100, and more.

Most MONAI Label applications are compatible with Linux and Windows operating systems, MacOS is not supported, however, some visualization tools can run on Mac,
users can set up MONAI Label server on a host machine, then use Mac for annotation (e.g., 3D Slicer).

## Trouble Shooting, Questions and Discussion

Ask and answer questions over
on [MONAI Label's GitHub Discussions tab](https://github.com/Project-MONAI/MONAILabel/discussions).

Join the conversation on Twitter [@ProjectMONAI](https://twitter.com/ProjectMONAI) or join
our [Slack channel](https://projectmonai.slack.com/archives/C031QRE0M1C).

## Cite

If you are using MONAI Label in your research, please use the following citation:

```bash
@article{DiazPinto2022monailabel,
   author = {Diaz-Pinto, Andres and Alle, Sachidanand and Ihsani, Alvin and Asad, Muhammad and
            Nath, Vishwesh and P{\'e}rez-Garc{\'\i}a, Fernando and Mehta, Pritesh and
            Li, Wenqi and Roth, Holger R. and Vercauteren, Tom and Xu, Daguang and
            Dogra, Prerna and Ourselin, Sebastien and Feng, Andrew and Cardoso, M. Jorge},
    title = {{MONAI Label: A framework for AI-assisted Interactive Labeling of 3D Medical Images}},
  journal = {arXiv e-prints},
     year = 2022,
     url  = {https://arxiv.org/pdf/2203.12362.pdf}
}
