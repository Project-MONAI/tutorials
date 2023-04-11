#!/bin/bash
NUM_GPUS_PER_NODE=${1}
gpu_memory=${2}
NUM_NODES=1

if [ ${NUM_GPUS_PER_NODE} -eq 2 ]
then
    export CUDA_VISIBLE_DEVICES=0,1
elif [ ${NUM_GPUS_PER_NODE} -eq 3 ]
then
    export CUDA_VISIBLE_DEVICES=0,1,2
elif [ ${NUM_GPUS_PER_NODE} -eq 4 ]
then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
elif [ ${NUM_GPUS_PER_NODE} -eq 1 ]
then
    export CUDA_VISIBLE_DEVICES=0
elif [ ${NUM_GPUS_PER_NODE} -eq 8 ]
then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

export NCCL_P2P_LEVEL=NVL
echo "CUDA:"$CUDA_VISIBLE_DEVICES

torchrun \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --nnodes=${NUM_NODES} \
    --master_addr=localhost --master_port=1234 \
    train_diffusion.py -c ./config/config_train_${gpu_memory}g.json -e ./config/environment.json -g ${NUM_GPUS_PER_NODE}
