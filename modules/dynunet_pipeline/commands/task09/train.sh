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

# train step 1, with large learning rate

lr=1e-2
fold=0

python train.py -fold $fold -train_num_workers 4 -interval 10 -num_samples 3 \
-learning_rate $lr -max_epochs 3000 -task_id 09 -pos_sample_num 2 \
-expr_name baseline -tta_val True -determinism_flag True -determinism_seed 0 \
-batch_dice True
