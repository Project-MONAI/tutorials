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

# During setup, stop on error
set -e

########################################################################
#                                                                      #
#                 append to this array if notebook                     #
#                 doesn't use the notion of epochs                     #
########################################################################
# These files don't loop across epochs
doesnt_contain_max_epochs=()
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" load_medical_images.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" integrate_3rd_party_transforms.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" transform_speed.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" transforms_demo_2d.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" nifti_read_example.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" 3d_image_transforms.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" mednist_classifier_ray.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" TorchIO_MONAI_PyTorch_Lightning.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" image_dataset.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" decollate_batch.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" csv_datasets.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" UNet_input_size_constrains.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" network_api.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" tcia_csv_processing.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" transform_visualization.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" 2d_inference_3d_volume.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" resample_benchmark.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" infoANDinference.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" dice_loss_metric_notes.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" 2d_slices_from_3d_sampling.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" 2d_slices_from_3d_training.ipynb)

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

doChecks=true
doRun=true
autofix=false
failfast=false
pattern="-and -name '*' -and ! -wholename '*federated_learning*'\
 -and ! -wholename '*transchex_openi*'\
 -and ! -wholename '*unetr_*'\
 -and ! -wholename '*profiling_camelyon*'\
 -and ! -wholename '*profiling_train_base_nvtx*'\
 -and ! -wholename '*benchmark_global_mutual_information*'\
 -and ! -wholename '*spleen_segmentation_3d_visualization_basic*'\
 -and ! -wholename '*deep_atlas_tutorial*'\
 -and ! -wholename '*nuclick_infer*'\
 -and ! -wholename '*nuclick_training_notebook*'\
 -and ! -wholename '*full_gpu_inference_pipeline*'\
 -and ! -wholename '*generate_random_permutations*'\
 -and ! -wholename '*get_started*'"
kernelspec="python3"

function print_usage {
    echo "runner.sh [--no-run] [--no-checks] [--autofix] [-f/--failfast] [-p/--pattern <find pattern>] [-h/--help]"
    echo            "[-v/--version]"
    echo ""
    echo "MONAI tutorials testing utilities. When running the notebooks, we first search for variables, such as"
    echo "\"max_epochs\" and set them to 1 to reduce testing time."
    echo ""
    echo "Code style check options:"
    echo "    --no-run          : don't run notebooks"
    echo "    --no-checks       : don't run code checks"
    echo "    --autofix         : autofix where possible"
    echo "    -f, --failfast    : stop on first error"
    echo "    -p, --pattern     : pattern of files to be run (added to \`find . -type f -name *.ipynb -and ! -wholename *.ipynb_checkpoints*\`)"
    echo "    -h, --help        : show this help message and exit"
    echo "    -v, --version     : show MONAI and system version information and exit"
    echo ""
    echo "Examples:"
    echo "./runner.sh                             # run full tests (${green}recommended before making pull requests${noColor})."
    echo "./runner.sh --no-run                    # don't run the notebooks."
    echo "./runner.sh --no-checks                 # don't run code checks."
    echo "./runner.sh --pattern \"-and \( -name '*read*' -or -name '*load*' \) -and ! -wholename '*acceleration*'\""
    echo "                                        # check filenames containing \"read\" or \"load\", but not if the"
    echo "                                          whole path contains \"deepgrow\"."
    echo "./runner.sh --kernelspec \"kernel\"       # Set the kernelspec value used to run notebooks, default is \"python3\"."
    echo ""
    echo "${separator}For bug reports, questions, and discussions, please file an issue at:"
    echo "    https://github.com/Project-MONAI/MONAI/issues/new/choose"
    echo ""
}

function print_style_fail_msg() {
    echo "${red}Check failed!${noColor}"
}
function print_error_msg() {
    echo "${red}Error: $1.${noColor}"
    echo ""
}

# parse arguments
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --no-run)
            doRun=false
        ;;
        --no-checks)
            doChecks=false
        ;;
        --autofix)
            autofix=true
        ;;
        -f|--failfast)
            failfast=true
        ;;
        -p|--pattern)
            pattern="$2"
            shift
        ;;
        -k|--kernelspec)
            kernelspec="$2"
            shift
        ;;
        -h|--help)
            print_usage
            exit 0
        ;;
        -v|--version)
            print_version
            exit 1
        ;;
        *)
            print_error_msg "Incorrect commandline provided, invalid key: $key"
            print_usage
            exit 1
        ;;
    esac
    shift
done

# if failfast, exit returning code. else increment number of failed tests and continue
function test_fail {
	print_style_fail_msg
	if [ $failfast = true ]; then
		exit $1
	fi
	current_test_successful=1
}

function check_installed {
	set +e
	command -v $1 &>/dev/null
	set -e
	success=$?
	if [ ${success} -ne 0 ]; then
		print_error_msg "Missing package: $1 (try pip install -r requirements.txt)"
		exit $success
	fi
}

# check that packages are installed
if [ $doRun = true ]; then
	check_installed papermill
fi
if [ $doChecks = true ]; then
	check_installed jupytext
	check_installed flake8
	if [ $autofix = true ]; then
		check_installed autopep8
		check_installed autoflake
	fi
fi


base_path="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd "${base_path}"

function replace_text {
	oldString="${s}\s*=\s*[0-9]\+"
	newString="${s} = 1"

	before=$(echo "$notebook" | grep "$oldString")
	[ ! -z "$before" ] && echo Before: && echo "$before"

	notebook=$(echo "$notebook" | sed "s/$oldString/$newString/g")

	after=$(echo "$notebook" | grep "$newString")
	[ ! -z "$after"  ] && echo After: && echo "$after"
}

# Get notebooks (pattern is "-and -name '*' -and ! -wholename '*federated_learning*'"
# unless user specifies otherwise)
files=($(echo $pattern | xargs find . -type f -name "*.ipynb" -and ! -wholename "*.ipynb_checkpoints*"))
if [[ $files == "" ]]; then
	print_error_msg "No files match pattern"
	exit 1
fi
echo "Files to be tested:"
for i in "${files[@]}"; do echo $i; done

# Keep track of number of passed tests. Use a trap to print results on exit
num_successful_tests=0
num_tested=0
# on finish
function finish {
  	if [[ ${num_successful_tests} -eq ${num_tested} ]]; then
		echo -e "\n\n\n${green}Testing finished. All ${num_tested} executed tests passed!${noColor}"
	else
		echo -e "\n\n\n${red}Testing finished. ${num_successful_tests} of ${num_tested} executed tests passed!${noColor}"
	fi
	# notification
	echo -e "\a"
	exit $((num_tested - num_successful_tests))
}
trap finish EXIT

# After setup, don't want to exit immediately after error
set +e

########################################################################
#                                                                      #
#  loop over files                                                     #
#                                                                      #
########################################################################
for file in "${files[@]}"; do
	current_test_successful=0

	echo "${separator}${blue}Running $file${noColor}"

	# Get to file's folder and get file contents
	path="$(dirname "${file}")"
	filename="$(basename "${file}")"
	cd ${base_path}/${path}

	########################################################################
	#                                                                      #
	#  code checks                                                         #
	#                                                                      #
	########################################################################
	if [ $doChecks = true ]; then

		if [ $autofix = true ]; then
			echo Applying autofixes...
			jupytext "$filename" --opt custom_cell_magics="writefile" \
				--pipe "autoflake --in-place --remove-unused-variables --imports numpy,monai,matplotlib,torch,ignite {}" \
				--pipe "autopep8 - --ignore W291 --max-line-length 120" \
				--pipe "sed 's/ = list()/ = []/'"
		fi

		# to check flake8, convert to python script, don't check
		# magic cells, and don't check line length for comment
		# lines (as this includes markdown), and then run flake8
		echo Checking PEP8 compliance...
		jupytext "$filename" --opt custom_cell_magics="writefile" -w --to script -o - | \
			sed 's/\(^\s*\)%/\1pass  # %/' | \
			sed 's/\(^#.*\)$/\1  # noqa: E501/' | \
			flake8 - --show-source --max-line-length 120
		success=$?
		if [ ${success} -ne 0 ]
	    then
	    	print_error_msg "Try running with autofixes: ${green}--autofix${noColor}"
	        test_fail ${success}
	    fi
	fi

	########################################################################
	#                                                                      #
	#  run notebooks with papermill                                        #
	#                                                                      #
	########################################################################
	if [ $doRun = true ]; then

		echo Running notebook...
		notebook=$(cat "$filename")

		# if compulsory keyword, max_epochs, missing...
		if [[ ! "$notebook" =~ "max_epochs" ]]; then
			# and notebook isn't in list of those expected to not have that keyword...
			should_contain_max_epochs=true
			for e in "${doesnt_contain_max_epochs[@]}"; do
				[[ "$e" == "$filename" ]] && should_contain_max_epochs=false && break
			done
			# then error
			if [[ $should_contain_max_epochs == true ]]; then
				print_error_msg "Couldn't find the keyword \"max_epochs\", and the notebook wasn't on the list of expected exemptions (\"doesnt_contain_max_epochs\")."
				test_fail 1
			fi
		fi

		# Set some variables to 1 to speed up proceedings
		strings_to_replace=(max_epochs val_interval disc_train_interval disc_train_steps num_batches_for_histogram)
		for s in "${strings_to_replace[@]}"; do
			replace_text
		done

		python -c 'import monai; monai.config.print_config()'
		time out=$(echo "$notebook" | papermill --progress-bar -k "$kernelspec")
		success=$?
	    if [[ ${success} -ne 0 || "$out" =~ "\"status\": \"failed\"" ]]; then
	        test_fail ${success}
	    fi
	fi

	num_tested=$((num_tested + 1))
	if [[ ${current_test_successful} -eq 0 ]]; then
		num_successful_tests=$((num_successful_tests + 1))
	fi
done
