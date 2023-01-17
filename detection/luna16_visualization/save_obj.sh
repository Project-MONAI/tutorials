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

INPUT_DATASET_JSON="./data_sample.json"
OUTPUT_DIR="./out_world_coord"

python save_obj.py  --input_box_mode "cccwhd" \
                    --input_dataset_json ${INPUT_DATASET_JSON} \
                    --output_dir ${OUTPUT_DIR}

IMAGE_DATA_ROOT="/data_root"
INPUT_DATASET_JSON="./data_sample_xyzxyz_image-coordinate.json"
OUTPUT_DIR="./out_image_coord"

python save_obj.py  --image_coordinate \
                    --image_data_root ${IMAGE_DATA_ROOT} \
                    --input_box_mode "xyzxyz" \
                    --input_dataset_json ${INPUT_DATASET_JSON} \
