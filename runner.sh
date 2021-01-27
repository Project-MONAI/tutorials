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

# Notification on finish
trap 'echo -e "\a"' EXIT

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

function print_usage {
    echo "runtests.sh [--no-run] [--no-checks] [--autofix] [--file <filename>] [--help] [--version]"
    echo ""
    echo "MONAI tutorials testing utilities. When running the notebooks, we first search for variables, such as `max_epochs` and set them to 1 to reduce testing time."
    echo ""
    echo "Examples:"
    echo "./runtests.sh                             # run full tests (${green}recommended before making pull requests${noColor})."
    echo "./runtests.sh --no-run                    # don't run the notebooks."
    echo "./runtests.sh --no-checks                 # don't run code checks."
    echo ""
    echo "Code style check options:"
    echo "    --no-run          : don't run notebooks"
    echo "    --no-checks       : don't run code checks"
    echo "    --autofix         : autofix where possible"
    echo "    --file            : only run on specified file(s). use as many times as desired"
    echo "    -h, --help        : show this help message and exit"
    echo "    -v, --version     : show MONAI and system version information and exit"
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

files=()

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
        --file)
            files=("${files[@]}" "$2")
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

function check_installed {
	set +e  # don't want immediate error
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
if [ $doChecks = true ]; then
	check_installed jupytext
	check_installed flake8
	if [ $autofix = true ]; then
		check_installed autopep8
		check_installed autoflake
	fi
fi


base_path="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

function replace_text {
	set +e  # otherwise blank grep causes error
	oldString="${s}\s*=\s*[0-9]\+"
	newString="${s} = 1"

	before=$(echo "$notebook" | grep "$oldString")
	[ ! -z "$before" ] && echo Before: && echo "$before"

	notebook=$(echo "$notebook" | sed "s/$oldString/$newString/g")

	after=$(echo "$notebook" | grep "$newString")
	[ ! -z "$after"  ] && echo After: && echo "$after"
	set -e
}

# If files haven't been added individually, get all
if [ -z ${files+x} ]; then
	files=($(find . -type f \( -name "*.ipynb" -and -not -iwholename "*.ipynb_checkpoints*" \)))
fi


########################################################################
#                                                                      #
#  loop over files                                                     #
#                                                                      #
########################################################################
for file in "${files[@]}"; do
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
			jupytext "$filename" \
				--pipe "autoflake --in-place --remove-unused-variables --imports numpy,monai,matplotlib,torch,ignite {}" \
				--pipe "autopep8 - --ignore W291" \
				--pipe "sed 's/ = list()/ = []/'"
		fi

		# to check flake8, convert to python script, don't check
		# magic cells, and don't check line length for comment
		# lines (as this includes markdown), and then run flake8
		echo Checking PEP8 compliance...
		jupytext "$filename" --to script -o - | \
			sed 's/\(^\s*\)%/\1pass  # %/' | \
			sed 's/\(^#.*\)$/\1  # noqa: E501/' | \
			flake8 - --show-source

		success=$?
		if [ ${success} -ne 0 ]
	    then
	    	echo "Try running with autofixes: ${green}--autofix${noColor}"
	        print_style_fail_msg
	        exit ${success}
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
	    if [[ ${success} -ne 0 || "$out" =~ "\"status\": \"failed\"" ]]; then
	        print_style_fail_msg
	        exit ${success}
	    fi
	fi

	echo "${green}passed!${noColor}"
done
