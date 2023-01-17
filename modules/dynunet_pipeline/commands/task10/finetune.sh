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
# since this task uses lr scheduler, please set the lr and max epochs
# here according to the step 1 training results. The value of max epochs equals
# to 2000 minus the best epoch in step 1.

lr=7.5e-3
fold=0
weight=model.pt

python train.py -fold $fold -train_num_workers 2 -interval 10 -num_samples 1 \
-learning_rate $lr -max_epochs 3000 -task_id 10 -pos_sample_num 1 \
-expr_name baseline -tta_val True -checkpoint $weight -determinism_flag True \
-determinism_seed 0 -lr_decay True
