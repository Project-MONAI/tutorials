#!/usr/bin/env bash
DATASET_DOWNLOAD_PATH="${projectpath}/data"
mkdir ${DATASET_DOWNLOAD_PATH}
python3 ${projectpath}/spleen_example/data/download_dataset.py -root_dir ${DATASET_DOWNLOAD_PATH}
echo "copy datalist files to ${DATASET_DOWNLOAD_PATH}/Task09_Spleen/."
cp ${projectpath}/spleen_example/data/dataset_*.json ${DATASET_DOWNLOAD_PATH}/Task09_Spleen/.
