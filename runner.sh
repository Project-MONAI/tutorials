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

# Exit on error
set -e

# TODO: replace this with:
# find . -type f \( -name "*.ipynb" -and -not -iwholename "*.ipynb_checkpoints*" \)
files=()
files=("${files[@]}" modules/load_medical_images.ipynb)
files=("${files[@]}" modules/autoencoder_mednist.ipynb)
files=("${files[@]}" modules/integrate_3rd_party_transforms.ipynb)
files=("${files[@]}" modules/3d_image_transforms.ipynb)

for file in "${files[@]}"; do
	echo "Running $file"

	# Get original notebook and set number of epochs to 1
	oldString="max_num_epochs\s*=\s*[0-9]\+"
	newString="max_num_epochs = 1"
	mod_notebook=$(cat "$file" | sed "s/$oldString/$newString/g")

	# Run with nbconvert
	out=$(echo "$mod_notebook" | jupyter nbconvert --execute --stdin --stdout --to notebook --ExecutePreprocessor.timeout=600)
	res=$?
	if [ $res -ne 0 ]; then
		echo -e $out
		exit $res
	fi
done
