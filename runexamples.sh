#!/bin/bash

# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
# script for running the examples

function setup() {
# install necessary packages
pip install numpy
pip install torch
pip install 'monai[itk, nibabel, pillow]'


# home directory
homedir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEMP_LOG="temp.txt"

cd "$homedir"
find "$homedir" -type f -name $TEMP_LOG -delete


# download data to specific directory
if [ -e "./testing_ixi_t1.tar.gz" ] && [ -d "./workspace/" ]; then
	echo "1" >> $TEMP_LOG
else
	wget  https://www.dropbox.com/s/y890gb6axzzqff5/testing_ixi_t1.tar.gz?dl=1
        mv testing_ixi_t1.tar.gz?dl=1 testing_ixi_t1.tar.gz
        mkdir -p ./workspace/data/medical/ixi/IXI-T1/
        tar -C ./workspace/data/medical/ixi/IXI-T1/ -xf testing_ixi_t1.tar.gz
fi
}

function 3d_class_torch() {
# run training files in 3d_classification/torch
for file in "3d_classification/torch"/*train*
do
    echo "Running $file"
    python "$file"
done

# check training files generated from 3d_classification/torch
[ -e "./best_metric_model_classification3d_array.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples 3d classification torch: model file not generated" | tee $TEMP_LOG && exit 0)
[ -e "./best_metric_model_classification3d_dict.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples 3d classification torch: model file not generated" | tee $TEMP_LOG && exit 0)

# run eval files in 3d_classification/torch
for file in "3d_classification/torch"/*eval*
do
    echo "Running $file"
    python "$file"
done
}

function 3d_class_ignite() {
# run training files in 3d_classification/ignite
for file in "3d_classification/ignite"/*train*
do
    echo "Running $file"
    python "$file"
done

# check training files generated from 3d_classification/ignite
[ -e "./runs_array/net_checkpoint_20.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples 3d classification ignite: model file not generated" | tee $TEMP_LOG && exit 0)
[ -e "./runs_dict/net_checkpoint_20.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples 3d classification ignite: model file not generated" | tee $TEMP_LOG && exit 0)

# run eval files in 3d_classification/ignite
for file in "3d_classification/ignite"/*eval*
do
    echo "Running $file"
    python "$file"
done
}

function 2d_seg_torch() {
# run training files in 2d_segmentation/torch
for file in "2d_segmentation/torch"/*train*
do
    echo "Running $file"
    python "$file"
done

# check training files generated from 2d_segmentation/torch
[ -e "./best_metric_model_segmentation2d_array.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples 2d segmentation torch: model file not generated" | tee $TEMP_LOG && exit 0)
[ -e "./best_metric_model_segmentation2d_dict.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples 2d segmentation torch: model file not generated" | tee $TEMP_LOG && exit 0)

# run eval files in 2d_segmentation/torch
for file in "2d_segmentation/torch"/*eval*
do
    python "$file"
done
}

function 3d_seg_torch() {
# run training files in 3d_segmentation/torch
for file in "3d_segmentation/torch"/*train*
do
    python "$file"
done

# check training files generated from 3d_segmentation/torch
[ -e "./best_metric_model_segmentation3d_array.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples 3d segmentation torch: model file not generated" | tee $TEMP_LOG && exit 0)
[ -e "./best_metric_model_segmentation3d_dict.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples 3d segmentation torch: model file not generated" | tee $TEMP_LOG && exit 0)

# run eval files in 3d_segmentation/torch
for file in "3d_segmentation/torch"/*eval*
do
    python "$file"
done

# run inference files in 3d_segmentation/torch
for file in "3d_segmentation/torch"/*inference*
do
    python "$file"
done
}

function 3d_seg_ignite() {
# run training files in 3d_segmentation/ignite
for file in "3d_segmentation/ignite"/*train*
do
    python "$file"
done

# check training files generated from 3d_segmentation/ignite
[ -e "./runs_array/net_checkpoint_100.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples 3d segmentation ignite: model file not generated" | tee $TEMP_LOG && exit 0)
[ -e "./runs_dict/net_checkpoint_50.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples 3d segmentation ignite: model file not generated" | tee $TEMP_LOG && exit 0)

# run eval files in 3d_segmentation/ignite
for file in "3d_segmentation/ignite"/*eval*
do
    python "$file"
done
}

# run training file in modules/workflows
function modules_workload() {
for file in "modules/workflows"/*train*
do
    echo "Running $file"
    python "$file"
done

# check training file generated from modules/workflows
[ -e "./runs/net_key_metric*.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples supervised workflows: model file not generated" | tee $TEMP_LOG && exit 0)
[ -e "./model_out/*.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples GAN workflows: model file not generated" | tee $TEMP_LOG && exit 0)

# run eval file in modules/workflows
for file in "modules/workflows"/*eval*
do
    python "$file"
done
}

# run the workloads
setup
3d_class_torch
3d_class_ignite
2d_seg_torch
3d_seg_torch
3d_seg_ignite
# TODO: there are no .py files. needs fix later
# modules_workload
