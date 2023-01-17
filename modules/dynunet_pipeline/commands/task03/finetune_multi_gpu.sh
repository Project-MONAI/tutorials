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

# train step 2, finetune with small learning rate
# please replace the weight variable into your actual weight

lr=1e-2
fold=0
weight=model.pt

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    --master_addr="localhost" --master_port=1234 \
    train.py -fold $fold -train_num_workers 8 -interval 20 -num_samples 1 \
    -learning_rate $lr -max_epochs 2000 -task_id 03 -pos_sample_num 1 \
    -expr_name baseline -tta_val True -checkpoint $weight -multi_gpu True \
    -eval_overlap 0.5 -sw_batch_size 2 -batch_dice True
