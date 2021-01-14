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

# Stop on error
set -e

# output formatting
separator=""
blue=""
green=""
red=""
noColor=""

if [[ -t 1 ]] # stdout is a terminal
then
    separator=$'--------------------------------------------------------------------------------\n'
    blue="$(tput bold; tput setaf 4)"
    green="$(tput bold; tput setaf 2)"
    red="$(tput bold; tput setaf 1)"
    noColor="$(tput sgr0)"
fi

doFlake8=true
doRun=true
autofix=false

function print_usage {
    echo "runtests.sh [--no-run] [--no-flake8] [--autofix] [--help] [--version]"
    echo ""
    echo "MONAI tutorials testing utilities."
    echo ""
    echo "Examples:"
    echo "./runtests.sh                             # run full tests (${green}recommended before making pull requests${noColor})."
    echo "./runtests.sh --no-run                    # don't run the notebooks."
    echo "./runtests.sh --no-flake8                 # don't run \"flake8\"."
    echo ""
    echo "Code style check options:"
    echo "    --no-run          : don't run notebooks"
    echo "    --no-flake8       : don't run \"flake8\" code format checks"
    echo "    --autofix         : autofix where possible"
    echo "    -h, --help        : show this help message and exit"
    echo "    -v, --version     : show MONAI and system version information and exit"
    echo ""
    echo "${separator}For bug reports, questions, and discussions, please file an issue at:"
    echo "    https://github.com/Project-MONAI/MONAI/issues/new/choose"
    echo ""
    exit 1
}

function print_version {
    ${cmdPrefix}${PY_EXE} -c 'import monai; monai.config.print_config()'
}

function print_style_fail_msg() {
    echo "${red}Check failed!${noColor}"
    echo "Try running with autofixes: ${green}--autofix${noColor}"
}

# parse arguments
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --no-run)
            doRun=false
        ;;
        --no-flake8)
            doFlake8=false
        ;;
        --autofix)
            autofix=true
        ;;
        -h|--help)
            print_usage
        ;;
        -v|--version)
            print_version
            exit 1
        ;;
        *)
            print_error_msg "Incorrect commandline provided, invalid key: $key"
            print_usage
        ;;
    esac
    shift
done

function check_installed {
	set +e
	command -v $1 &>/dev/null
	success=$?
	if [ ${success} -ne 0 ]; then
		echo "${red}Missing package: $1 (try pip install -r requirements.txt)${noColor}"
		exit $success
	fi
	set -e
}

# check that packages are installed
if [ $doRun = true ]; then
	check_installed papermill
fi
if [ $doFlake8 = true ]; then
	check_installed jupytext
	check_installed flake8
	if [ $autofix = true ]; then
		check_installed autopep8
	fi
fi


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
files=("${files[@]}" modules/load_medical_images.ipynb)
files=("${files[@]}" modules/autoencoder_mednist.ipynb)
files=("${files[@]}" modules/integrate_3rd_party_transforms.ipynb)
files=("${files[@]}" modules/transforms_demo_2d.ipynb)
files=("${files[@]}" modules/nifti_read_example.ipynb)
files=("${files[@]}" modules/post_transforms.ipynb)
files=("${files[@]}" modules/3d_image_transforms.ipynb)
files=("${files[@]}" modules/public_datasets.ipynb)
files=("${files[@]}" modules/varautoencoder_mednist.ipynb)
files=("${files[@]}" modules/models_ensemble.ipynb)
files=("${files[@]}" modules/layer_wise_learning_rate.ipynb)
files=("${files[@]}" modules/mednist_GAN_tutorial.ipynb)
files=("${files[@]}" modules/mednist_GAN_workflow_array.ipynb)
files=("${files[@]}" modules/mednist_GAN_workflow_dict.ipynb)
files=("${files[@]}" 2d_classification/mednist_tutorial.ipynb)
files=("${files[@]}" 3d_classification/torch/densenet_training_array.ipynb)
files=("${files[@]}" 3d_segmentation/spleen_segmentation_3d.ipynb)
files=("${files[@]}" 3d_segmentation/spleen_segmentation_3d_lightning.ipynb)
files=("${files[@]}" acceleration/transform_speed.ipynb)

# Currently testing
# files=("${files[@]}" acceleration/automatic_mixed_precision.ipynb)

# Tested -- requires update
# files=("${files[@]}" modules/dynunet_tutorial.ipynb)

# Not tested
# files=("${files[@]}" 3d_segmentation/brats_segmentation_3d.ipynb)
# files=("${files[@]}" 3d_segmentation/unet_segmentation_3d_catalyst.ipynb)
# files=("${files[@]}" 3d_segmentation/unet_segmentation_3d_ignite.ipynb)
# files=("${files[@]}" acceleration/dataset_type_performance.ipynb)
# files=("${files[@]}" acceleration/fast_training_tutorial.ipynb)
# files=("${files[@]}" acceleration/multi_gpu_test.ipynb)
# files=("${files[@]}" acceleration/threadbuffer_performance.ipynb)
# files=("${files[@]}" modules/interpretability/class_lung_lesion.ipynb)

# These files don't loop across epochs
doesnt_contain_max_epochs=()
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" modules/load_medical_images.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" modules/integrate_3rd_party_transforms.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" acceleration/transform_speed.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" modules/transforms_demo_2d.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" modules/nifti_read_example.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" modules/3d_image_transforms.ipynb)

contains_element () {
  local e match="$1"
  shift
  contained=false
  for e; do [[ "$e" == "$match" ]] && contained=true; done
  echo $contained
}

for file in "${files[@]}"; do
	echo "${separator}${blue}Running $file${noColor}"

	# Get to file's folder and get file contents
	path="$(dirname "${file}")"
	filename="$(basename "${file}")"
	cd ${base_path}/${path}

	########################################################################
	#                                                                      #
	#  flake8                                                              #
	#                                                                      #
	########################################################################
	if [ $doFlake8 = true ]; then

		if [ $autofix = true ]; then
			jupytext "$filename" --pipe "autopep8"
		fi
		
		# to check flake8, convert to python script, don't check magic cells,
		# indented import monai (to check if installed), or line length for
		# comment lines (as this includes markdown), and then run flake8
		jupytext "$filename" --to script -o - | \
			sed 's/\(^\s*\)%/\1pass  # %/' | \
			sed 's/    import monai/    pass/' | \
			sed 's/\(^#.*\)$/\1  # noqa: E501/' | \
			flake8 - --show-source

		success=$?
		if [ ${success} -ne 0 ]
	    then
	        print_style_fail_msg
	        exit ${success}
	    else
	        echo "${green}passed!${noColor}"
	    fi
	fi

	########################################################################
	#                                                                      #
	#  run notebooks with papermill                                        #
	#                                                                      #
	########################################################################
	if [ $doRun = true ]; then

		notebook=$(cat "$filename")

		# check that it contains the keyword max_epochs (unless expected not to)
		if [[ ! "$notebook" =~ "max_epochs" ]]; then
			if [[ $(contains_element "$file" "${doesnt_contain_max_epochs[@]}") == false ]]; then
				echo "Couldn't find the keyword \"max_epochs\", and the notebook wasn't on the list of expected exemptions (\"doesnt_contain_max_epochs\")."
				print_style_fail_msg
				exit 1
			fi
		fi

		# Set some variables to 1 to speed up proceedings
		strings_to_replace=(max_epochs val_interval disc_train_interval disc_train_steps)
		for s in "${strings_to_replace[@]}"; do
			replace_text
		done
		
		# echo "$notebook" > "${base_path}/debug_notebook.ipynb"
		out=$(echo "$notebook" | papermill --progress-bar)
		success=$?
	    if [ ${success} -ne 0 ]
	    then
	        print_style_fail_msg
	        exit ${success}
	    else
	        echo "${green}passed!${noColor}"
	    fi
	fi
done
