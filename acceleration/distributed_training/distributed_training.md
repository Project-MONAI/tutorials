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

Let's take a two-node (16 GPUs in total) model training example. In the primary node (node rank 0), we run the following command.

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
