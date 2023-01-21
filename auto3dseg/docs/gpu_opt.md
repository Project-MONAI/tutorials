## GPU Utlization Optimization

### Introduction
We introduced an automated solution to optimize the GPU usage of algorithms in Auto3DSeg.
Typically, the most time-consuming process in Auto3DSeg is model training.
Sometimes the low GPU utilization is because that GPU capacities is not fully utilized with fixed hyperparameters.
Our proposed solution is capable to automatically estimate hyper-parameters in model training configurations maximizing utilities of the available GPU capacities.
The solution is leveraging hyper-parameter optimization algorithms to search for optimital hyper-parameters with any given GPU devices.

The following hyper-paramters in model training configurations are optimized in the process.

1. **num_images_per_batch:** Batch size determines how many images are in each mini-batch and how many training iterations per epoch. Large batch size can reduce training time per epoch and increase GPU memory usage with decent CPU capacities for I/O;
2. **num_sw_batch_size:** Batch size in sliding-window inference directly relates to how many patches are in one pass of model feedforward operation. Large batch size in sliding-window inference can reduce overall inference time and increase GPU memory usage;
3. **validation_data_device:** Validation device indicates if the volume is stored on GPU or CPU. Ideally, it would be fast to store input volumes onto GPU for inference. However, if 3D volumes are very large and GPU memory is limited, we have to store the image arrays on CPU (instead of GPU) and put patches of volumes on GPU for inference;
4. **num_trials:** The trial number defines the time length of the optimization process. The larger the number of trials, the longer the optimization process.

### Usage

### Effect
