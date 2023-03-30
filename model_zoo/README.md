This folder contains tutorials and examples to show how to use an existing bundle of our released models, including:

- Adapt a bundle to another dataset.
- Integrate a bundle into your own application.
- Load the pre-trained weights from a bundle and do transfer-learning.
- Extend the features of workflow in MONAI bundles based on `event-handler` mechanism.

## Getting Started

To download and get started with the models, `monai.bundle` API is recommended. The following is an example to download the `spleen_ct_segmentation` bundle:

```shell
python -m monai.bundle download --name spleen_ct_segmentation --bundle_dir "./"
```
