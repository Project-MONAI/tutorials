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

base_path="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

function replace_text {
	oldString="${s}\s*=\s*[0-9]\+"
	newString="${s} = 1"

	before=$(echo "$notebook" | grep "$oldString")
	[ ! -z "$before" ] && echo Before: && echo "$before"

	notebook=$(echo "$notebook" | sed "s/$oldString/$newString/g")
	
	after=$(echo "$notebook" | grep "$newString")
	[ ! -z "$after"  ] && echo After: && echo "$after"
}

# TODO: replace this with:
# find . -type f \( -name "*.ipynb" -and -not -iwholename "*.ipynb_checkpoints*" \)
files=()

# Tested -- working
# files=("${files[@]}" modules/load_medical_images.ipynb)
# files=("${files[@]}" modules/autoencoder_mednist.ipynb)
# files=("${files[@]}" modules/integrate_3rd_party_transforms.ipynb)
# files=("${files[@]}" modules/transforms_demo_2d.ipynb)
# files=("${files[@]}" modules/nifti_read_example.ipynb)
# files=("${files[@]}" modules/post_transforms.ipynb)
# files=("${files[@]}" modules/3d_image_transforms.ipynb)
# files=("${files[@]}" modules/public_datasets.ipynb)
# files=("${files[@]}" modules/varautoencoder_mednist.ipynb)
# files=("${files[@]}" modules/models_ensemble.ipynb)
# files=("${files[@]}" modules/layer_wise_learning_rate.ipynb)
# files=("${files[@]}" modules/mednist_GAN_tutorial.ipynb)
# files=("${files[@]}" modules/mednist_GAN_workflow_array.ipynb)
# files=("${files[@]}" modules/mednist_GAN_workflow_dict.ipynb)
# files=("${files[@]}" 2d_classification/mednist_tutorial.ipynb)

# Currently testing
files=("${files[@]}" 3d_classification/torch/densenet_training_array.ipynb)

# Tested -- requires update
# files=("${files[@]}" modules/dynunet_tutorial.ipynb)

# Not tested
# files=("${files[@]}" 3d_segmentation/brats_segmentation_3d.ipynb)
# files=("${files[@]}" 3d_segmentation/spleen_segmentation_3d.ipynb)
# files=("${files[@]}" 3d_segmentation/spleen_segmentation_3d_lightning.ipynb)
# files=("${files[@]}" 3d_segmentation/unet_segmentation_3d_catalyst.ipynb)
# files=("${files[@]}" 3d_segmentation/unet_segmentation_3d_ignite.ipynb)
# files=("${files[@]}" acceleration/automatic_mixed_precision.ipynb)
# files=("${files[@]}" acceleration/dataset_type_performance.ipynb)
# files=("${files[@]}" acceleration/fast_training_tutorial.ipynb)
# files=("${files[@]}" acceleration/multi_gpu_test.ipynb)
# files=("${files[@]}" acceleration/threadbuffer_performance.ipynb)
# files=("${files[@]}" acceleration/transform_speed.ipynb)
# files=("${files[@]}" modules/interpretability/class_lung_lesion.ipynb)

for file in "${files[@]}"; do
	echo -e "\nRunning $file"

	# Get to file's folder and get file contents
	path="$(dirname "${file}")"
	filename="$(basename "${file}")"
	cd ${base_path}/${path}
	notebook=$(cat "$filename")

	# Set some variables to 1 to speed up proceedings
	strings_to_replace=(max_num_epochs max_epochs val_interval disc_train_interval disc_train_steps)
	for s in "${strings_to_replace[@]}"; do
		replace_text
	done

	# Run with nbconvert
	# echo "$notebook" > "${base_path}/debug_notebook.ipynb"
	out=$(echo "$notebook" | papermill --progress-bar)
	res=$?
	if [ $res -ne 0 ]; then
		echo -e $out
		exit $res
	fi
done
