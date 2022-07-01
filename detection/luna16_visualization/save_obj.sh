#!/bin/bash

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
