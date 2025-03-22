## Data Preparation

Users need to download the dataset Task01_BrainTumour from the MICCAI challenge [Medical Segmentation Decathlon](http://medicaldecathlon.com/) for the following examples.
- Define the directory where to download and extract the dataset
```
root_dir="/path/to/your/directory"  # Change this to your desired directory
```
- Download and extract in one line if the directory doesn't exist
```
[ ! -d "${root_dir}/Task01_BrainTumour" ] && mkdir -p "${root_dir}/Task01_BrainTumour" && wget -qO- "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar" | tar -xv -C "${root_dir}"
```

## Multi-GPU Training

Users can set your `NUM_GPUS_PER_NODE`, `NUM_NODES`, `INDEX_CURRENT_NODE`, as well as `DIR_OF_DATA` for the directory of the test dataset.
Then users can execute the following command to start multi-GPU model training:

```
torchrun --nproc_per_node=NUM_GPUS_PER_NODE --nnodes=NUM_NODES brats_training_ddp.py -d DIR_OF_DATA
```

## Multi-Node Training

Let's take two-node (16 GPUs in total) model training as an example. In the primary node (node rank 0), we run the following command.

```
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=PRIMARY_NODE_IP --master_port=1234 brats_training_ddp.py
```
Here, `PRIMARY_NODE_IP` is the IP address of the first node.

In the second node (node rank 1), we run the following command.

```
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="localhost" --master_port=1234 brats_training_ddp.py
```

Note that the only difference between the two commands is `--node_rank`.

There would be some possible delay between the execution of the two commands in the two nodes. But the first node would always wait for the second one, and they would start and train together. If there is an IP issue for the validation part during model training, please refer to the solution [here](https://discuss.pytorch.org/t/connect-127-0-1-1-a-port-connection-refused/100802/25) to resolve it.

## Running on MacBook M4 Max 2024

To run a model on a MacBook M4 Max 2024, follow these steps to optimize GPU utilization:

1. **Install Dependencies**:
   - Install Homebrew if you haven't already:
     ```sh
     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
     ```
   - Install Python:
     ```sh
     brew install python
     ```
   - Install virtualenv:
     ```sh
     pip install virtualenv
     ```
   - Create a virtual environment:
     ```sh
     virtualenv monai_env
     ```
   - Activate the virtual environment:
     ```sh
     source monai_env/bin/activate
     ```
   - Install MONAI and other dependencies:
     ```sh
     pip install monai numpy torch torchvision
     ```

2. **Set Up the Environment**:
   - Clone the Project-MONAI repository:
     ```sh
     git clone https://github.com/Project-MONAI/tutorials.git
     cd tutorials
     ```
   - Navigate to the desired tutorial directory, for example:
     ```sh
     cd acceleration
     ```

3. **Run a Model**:
   - Choose the tutorial or example you want to run. For instance, to run the `fast_training_tutorial.ipynb`, you can use Jupyter Notebook.
   - Install Jupyter Notebook:
     ```sh
     pip install notebook
     ```
   - Start Jupyter Notebook:
     ```sh
     jupyter notebook
     ```
   - Open the desired notebook (e.g., `fast_training_tutorial.ipynb`) in your browser and follow the instructions to run the model.

By following these steps, you should be able to install and run a model on your MacBook M4 Max 2024 with the specified system information.
