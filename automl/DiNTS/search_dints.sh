#!/bin/bash
clear

TASK="Task07_Pancreas"

DATA_ROOT="/workspace/data_msd/${TASK}"
JSON_PATH="${DATA_ROOT}/dataset.json"

FOLD=4
NUM_FOLDS=5

NUM_GPUS_PER_NODE=4
NUM_NODES=1

if [ ${NUM_GPUS_PER_NODE} -eq 1 ]
then
    export CUDA_VISIBLE_DEVICES=0
elif [ ${NUM_GPUS_PER_NODE} -eq 2 ]
then
    export CUDA_VISIBLE_DEVICES=0,1
elif [ ${NUM_GPUS_PER_NODE} -eq 4 ]
then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
elif [ ${NUM_GPUS_PER_NODE} -eq 8 ]
then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

CHECKPOINT_ROOT="models/search_${TASK}_fold${FOLD}"
CHECKPOINT="${CHECKPOINT_ROOT}/best_metric_model.pth"
FACTOR_RAM_COST=0.8
JSON_KEY="training"
OUTPUT_ROOT="models/search_${TASK}_fold${FOLD}_ram${FACTOR_RAM_COST}"

python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --nnodes=${NUM_NODES} \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=1234 \
    search_dints.py --checkpoint=${CHECKPOINT} \
                    --factor_ram_cost=${FACTOR_RAM_COST} \
                    --fold=${FOLD} \
                    --json=${JSON_PATH} \
                    --json_key=${JSON_KEY} \
                    --num_folds=${NUM_FOLDS} \
                    --output_root=${OUTPUT_ROOT} \
                    --root=${DATA_ROOT}
