#!/bin/bash

INPUT_DIR="./tmp"
INPUT_DATASET_JSON="./tmp/dataset_synthetic.json"
OUTPUT_DIR="./tmp_out"

python save_obj.py  --input_dir ${INPUT_DIR} \
                    --input_dataset_json ${INPUT_DATASET_JSON} \
                    --lpi_to_ras \
                    --output_dir ${OUTPUT_DIR} \
