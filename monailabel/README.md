# MONAI Label End-To-End Tutorial Series

This folder contains end-to-end tutorials of MONAI Label applications. Sample apps including `radiology`, `pathology`, `endoscopy` and `monaibundle`.

![image](https://raw.githubusercontent.com/Project-MONAI/MONAILabel/main/docs/images/sampleApps_index.jpeg)

These tutorials demonstrate step-by-step installations and operations of MONAI Label server, client viewers, or datastore platforms.

## Content

- Radiology App:
  - Viewer: 3D Slicer | Datastore: Local | Task: Segmentation
    - [MONAILabel: HelloWorld_radiology_3dslicer](monailabel_HelloWorld_radiology_3dslicer.ipynb): Spleen segmentation with 3D Slicer setups.
- MONAIBUNDLE App:
  - Viewer: 3D Slicer | Datastore: Local | Task: Segmentation
    - [MONAILabel: pancreas_tumor_segmentation_3DSlicer](monailabel_pancreas_tumor_segmentation_3DSlicer.ipynb): Pancreas and tumor segmentation with CT scans in 3D Slicer.
    - [MONAILabel: monaibundle_3dslicer_multiorgan_seg](monailabel_monaibundle_3dslicer_multiorgan_seg.ipynb): Multi-organ segmentation with CT scans in 3D Slicer.
- Pathology App:
  - Viewer: QuPath | Datastore: Local | Task: Segmentation
    - [MONAILabel: pathology_nuclei_segmentation_QuPath](monailabel_pathology_nuclei_segmentation_QuPath.ipynb) Nuclei segmentation with QuPath setup and Nuclick models.
- Endoscopy App:
  - Viewer: CVAT | Datastore: Local | Task: Segmentation
    - [MONAILabel: endoscopy_cvat_tooltracking](monailabel_endoscopy_cvat_tooltracking.ipynb): Surgical tool segmentation with CVAT/Nuclio setup.
