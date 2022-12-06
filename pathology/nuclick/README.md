# Nuclick and Classification Examples
This folder contains examples to run train and validate a nuclick and classification models.
It also has notebooks to run inference over trained (monai-zoo) model.

### 1. Data

Training these model requires data. Some public available datasets which are used in the examples can be downloaded from [ConSeP](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet).

### 2. Questions and bugs

- For questions relating to the use of MONAI, please us our [Discussions tab](https://github.com/Project-MONAI/MONAI/discussions) on the main repository of MONAI.
- For bugs relating to MONAI functionality, please create an issue on the [main repository](https://github.com/Project-MONAI/MONAI/issues).
- For bugs relating to the running of a tutorial, please create an issue in [this repository](https://github.com/Project-MONAI/Tutorials/issues).

### 3. List of notebooks and examples

#### NuClick Interaction Model
#### [Training](./nuclick_training_notebook.ipynb)
This notebook guides to train a NuClick model.

#### [Inference](./nuclick_infer.ipynb)
This notebook guides to segment a nuclei for given user click(s).

#### Nuclei Classification Model
#### [Training](./nuclei_classification_training_notebook.ipynb)
This notebook guides to train a classification model for certain nuclei types.

#### [Inference](./nuclei_classification_infer.ipynb)
This notebook helps to infer classification model for a given Nulei.
