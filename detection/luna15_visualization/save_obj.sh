#!/bin/bash

INPUT_DATASET_JSON="./data_sample.json"
OUTPUT_DIR="./out"

python save_obj.py  --input_dataset_json ${INPUT_DATASET_JSON} \
                    --output_dir ${OUTPUT_DIR}
