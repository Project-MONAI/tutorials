# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

# python3 train_autoencoder.py -c ./config/config_train_${gpu_memory}g.json -g 1
torchrun \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --nnodes=${NUM_NODES} \
    --master_addr=localhost --master_port=1234 \
    train_autoencoder.py -c ./config/config_train_${gpu_memory}g.json -e ./config/environment.json -g ${NUM_GPUS_PER_NODE}
