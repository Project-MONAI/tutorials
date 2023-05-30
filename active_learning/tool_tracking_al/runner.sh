#!/bin/bash

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

DATA_ROOT="/to/be/defined"
JSON_PATH="/to/be/defined"
LOG_DIR="/to/be/defined"

# Full dataset baseline
python active_learning.py --base_dir ${LOG_DIR}/all_data_iter5k --data_root ${DATA_ROOT} --json_path ${JSON_PATH} --seed 240 --active_iters 1 --dropout_ratio 0.2 --mc_number 10 --initial_pool 360 --queries 0 --strategy random --steps 5000 --val_interval 1 --batch_size 1 --val_batch_size 1 --lr 1e-4

# Initial Pool 20 Queries 20 Random Strategy
python active_learning.py --base_dir ${LOG_DIR}/random_i15_q15_iter5k --data_root ${DATA_ROOT} --json_path ${JSON_PATH} --seed 240 --active_iters 5 --dropout_ratio 0.2 --mc_number 10 --initial_pool 20 --queries 20 --strategy random --steps 5000 --val_interval 1 --batch_size 1 --val_batch_size 1 --lr 1e-4

# Initial Pool 20 Queries 20 Variance Strategy
python active_learning.py --base_dir ${LOG_DIR}/variance_i15_q15_iter5k --data_root ${DATA_ROOT} --json_path ${JSON_PATH} --seed 240 --active_iters 5 --dropout_ratio 0.2 --mc_number 10 --initial_pool 20 --queries 20 --strategy random --steps 5000 --val_interval 1 --batch_size 1 --val_batch_size 1 --lr 1e-4
