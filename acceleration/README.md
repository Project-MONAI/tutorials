# Performance optimization and GPU acceleration
Typically, model training is a time-consuming step during deep learning development, especially in medical imaging applications. Volumetric medical images are usually large (as multi-dimensional arrays) and the model training process can be complex. Even with powerful hardware (e.g. CPU/GPU with large RAM), it is not easy to fully leverage them to achieve high performance. NVIDIA GPUs have been widely applied in many areas of deep learning training and evaluation, and the CUDA parallel computation shows obvious acceleration when comparing to traditional computation methods. To fully leverage GPU features, many popular mechanisms raised, like automatic mixed precision (AMP), distributed data parallel, etc. MONAI can support these features and this folder provides a fast training guide to achieve the best performance and rich examples.

### List of notebooks and examples
#### [fast_model_training_guide](./fast_model_training_guide.md)
The document introduces details of how to profile the training pipeline, how to analyze the dataset and select suitable algorithms, and how to optimize GPU utilization in single GPU, multi-GPUs or even multi-nodes.
#### [distributed_training](./distributed_training)
The examples show how to execute distributed training and evaluation based on 3 different frameworks:
- PyTorch native `DistributedDataParallel` module with `torchrun`.
- Horovod APIs with `horovodrun`.
- PyTorch ignite and MONAI workflows.

They can run on several distributed nodes with multiple GPU devices on every node.
#### [automatic_mixed_precision](./automatic_mixed_precision.ipynb)
And compares the training speed and memory usage with/without AMP.
#### [dataset_type_performance](./dataset_type_performance.ipynb)
This notebook compares the performance of `Dataset`, `CacheDataset` and `PersistentDataset`. These classes differ in how data is stored (in memory or on disk), and at which moment transforms are applied.
#### [fast_training_tutorial](./fast_training_tutorial.ipynb)
This tutorial compares the training performance of pure PyTorch program and optimized program in MONAI based on NVIDIA GPU device and latest CUDA library.
The optimization methods mainly include: `AMP`, `CacheDataset` and `Novograd`.
#### [threadbuffer_performance](./threadbuffer_performance.ipynb)
Demonstrates the use of the `ThreadBuffer` class used to generate data batches during training in a separate thread.
#### [transform_speed](./transform_speed.ipynb)
Illustrate reading NIfTI files and test speed of different transforms on different devices.
#### [TensorRT_inference_acceleration](./TensorRT_inference_acceleration.ipynb)
This notebook shows how to use TensorRT to accelerate the model and achieve a better inference latency.

#### [Tutorials for resource monitoring](./monitoring/README.md)
Information about how to set up and apply existing tools to monitor the computing resources.

### Running a Model on MacBook M4 Max 2024

#### Step-by-Step Guide

##### 1. Installing Dependencies

To run a model on a MacBook M4 Max 2024, you need to install the necessary dependencies. Follow these steps:

1. Install Homebrew if you haven't already:
   ```sh
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Install Python:
   ```sh
   brew install python
   ```

3. Install virtualenv:
   ```sh
   pip install virtualenv
   ```

4. Create a virtual environment:
   ```sh
   virtualenv monai_env
   ```

5. Activate the virtual environment:
   ```sh
   source monai_env/bin/activate
   ```

6. Install MONAI and other dependencies:
   ```sh
   pip install monai numpy torch torchvision
   ```

##### 2. Setting Up the Environment

1. Clone the Project-MONAI repository:
   ```sh
   git clone https://github.com/Project-MONAI/tutorials.git
   cd tutorials
   ```

2. Navigate to the desired tutorial directory, for example:
   ```sh
   cd acceleration
   ```

##### 3. Running a Model

1. Choose the tutorial or example you want to run. For instance, to run the `fast_training_tutorial.ipynb`, you can use Jupyter Notebook.

2. Install Jupyter Notebook:
   ```sh
   pip install notebook
   ```

3. Start Jupyter Notebook:
   ```sh
   jupyter notebook
   ```

4. Open the desired notebook (e.g., `fast_training_tutorial.ipynb`) in your browser and follow the instructions to run the model.

By following these steps, you should be able to install and run a model on your MacBook M4 Max 2024 with the specified system information.
