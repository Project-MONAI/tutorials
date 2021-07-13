#!/usr/bin/env bash

DATASET_DOWNLOAD_PATH="${projectpath}/../data"
python3 ${projectpath}/../data/download_dataset.py -root_dir ${DATASET_DOWNLOAD_PATH}
echo "copy datalist files to ${DATASET_DOWNLOAD_PATH}/Task09_Spleen/."
cp ${projectpath}/../data/dataset_*.json ${DATASET_DOWNLOAD_PATH}/Task09_Spleen/.
