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

# Get original notebook and set number of epochs to 1
orig_notebook=$(cat 2d_classification/mednist_tutorial.ipynb)
oldString="max_num_epochs\s*=\s*[0-9]\+"
newString="max_num_epochs = 1"
mod_notebook=$(echo "$orig_notebook" | sed "s/$oldString/$newString/g")

# Run with nbconvert
echo "$mod_notebook" > temp.ipynb

echo "$mod_notebook" | jupyter nbconvert --execute --stdin --stdout --to notebook
