#!/bin/bash

# Copyright (c) MONAI Consortium
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
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" msd_datalist_generator.ipynb)
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
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" data_analyzer.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" data_analyzer_byoc.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" transforms_metatensor.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" 2d_slices_from_3d_training.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" preprocess_to_build_detection_dataset.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" preprocess_detect_scene_and_split_fold.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" preprocess_extract_images_from_video.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" profiling_camelyon_pipeline.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" monailabel_HelloWorld_radiology_3dslicer.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" monailabel_monaibundle_3dslicer_multiorgan_seg.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" monailabel_bring_your_own_data.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" monailabel_endoscopy_cvat_tooltracking.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" monailabel_pathology_nuclei_segmentation_QuPath.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" monailabel_radiology_spleen_segmentation_OHIF.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" monailabel_wholebody_totalSegmentator_3dslicer.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" monailabel_monaibundle_3dslicer_lung_nodule_detection.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" monailabel_pathology_HoVerNet_QuPath.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" example_feature.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" ssl_train.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" ssl_finetune.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" swinunetr_finetune.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" results_uncertainty_analysis.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" TCIA_PROSTATEx_Prostate_MRI_Anatomy_Model.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" lazy_resampling_functional.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" lazy_resampling_compose.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" TensorRT_inference_acceleration.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" lazy_resampling_benchmark.ipynb)
doesnt_contain_max_epochs=("${doesnt_contain_max_epochs[@]}" modular_patch_inferer.ipynb)

# Execution of the notebook in these folders / with the filename cannot be automated
skip_run_papermill=()
skip_run_papermill=("${skip_run_papermill[@]}" .*federated_learning*)
skip_run_papermill=("${skip_run_papermill[@]}" .*transchex_openi*)
skip_run_papermill=("${skip_run_papermill[@]}" .*unetr_*)
skip_run_papermill=("${skip_run_papermill[@]}" .*profiling_train_base_nvtx*)
skip_run_papermill=("${skip_run_papermill[@]}" .*benchmark_global_mutual_information*)
skip_run_papermill=("${skip_run_papermill[@]}" .*spleen_segmentation_3d_visualization_basic*)
skip_run_papermill=("${skip_run_papermill[@]}" .*full_gpu_inference_pipeline*)
skip_run_papermill=("${skip_run_papermill[@]}" .*generate_random_permutations*)
skip_run_papermill=("${skip_run_papermill[@]}" .*transforms_update_meta_data*)
skip_run_papermill=("${skip_run_papermill[@]}" .*video_seg*)
skip_run_papermill=("${skip_run_papermill[@]}" .*tcia_dataset*)
skip_run_papermill=("${skip_run_papermill[@]}" .*hovernet_torch*)
skip_run_papermill=("${skip_run_papermill[@]}" .*preprocess_detect_scene_and_split_fold*)
skip_run_papermill=("${skip_run_papermill[@]}" .*preprocess_to_build_detection_dataset*)
skip_run_papermill=("${skip_run_papermill[@]}" .*preprocess_extract_images_from_video*)
skip_run_papermill=("${skip_run_papermill[@]}" .*transfer_mmar*)
skip_run_papermill=("${skip_run_papermill[@]}" .*MRI_reconstruction*)
skip_run_papermill=("${skip_run_papermill[@]}" .*monailabel_HelloWorld_radiology_3dslicer*)
skip_run_papermill=("${skip_run_papermill[@]}" .*monailabel_monaibundle_3dslicer_multiorgan_seg*)
skip_run_papermill=("${skip_run_papermill[@]}" .*monailabel_bring_your_own_data*)
skip_run_papermill=("${skip_run_papermill[@]}" .*monailabel_endoscopy_cvat_tooltracking*)
skip_run_papermill=("${skip_run_papermill[@]}" .*monailabel_pathology_nuclei_segmentation_QuPath*)
skip_run_papermill=("${skip_run_papermill[@]}" .*monailabel_radiology_spleen_segmentation_OHIF*)
skip_run_papermill=("${skip_run_papermill[@]}" .*monailabel_wholebody_totalSegmentator_3dslicer*)
skip_run_papermill=("${skip_run_papermill[@]}" .*monailabel_monaibundle_3dslicer_lung_nodule_detection*)
skip_run_papermill=("${skip_run_papermill[@]}" .*monailabel_pathology_HoVerNet_QuPath*)
skip_run_papermill=("${skip_run_papermill[@]}" .*ssl_train*)
skip_run_papermill=("${skip_run_papermill[@]}" .*ssl_finetune*)
skip_run_papermill=("${skip_run_papermill[@]}" .*swinunetr_finetune*)
skip_run_papermill=("${skip_run_papermill[@]}" .*active_learning*)
skip_run_papermill=("${skip_run_papermill[@]}" .*transform_visualization*)  # https://github.com/Project-MONAI/tutorials/issues/1155
skip_run_papermill=("${skip_run_papermill[@]}" .*TensorRT_inference_acceleration*)
skip_run_papermill=("${skip_run_papermill[@]}" .*mednist_classifier_ray*)  # https://github.com/Project-MONAI/tutorials/issues/1307
skip_run_papermill=("${skip_run_papermill[@]}" .*TorchIO_MONAI_PyTorch_Lightning*)  # https://github.com/Project-MONAI/tutorials/issues/1324

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
doCopyright=false
doRun=true
doStandardizeCells=false
autofix=false
failfast=false
pattern=""
papermill_opt=""

kernelspec="python3"

PY_EXE=${MONAI_PY_EXE:-$(which python)}
NB_TEST="$PY_EXE ci/nbtest.py"
NB_OUTPUT_LINE_CAP=100

function print_usage {
    echo "runner.sh [--no-run] [--no-checks] [--autofix] [-f/--failfast] [-p/--pattern <find pattern>] [-h/--help]"
    echo            "[-v/--version] [--verbose]"
    echo ""
    echo "MONAI tutorials testing utilities. When running the notebooks, we first search for variables, such as"
    echo "\"max_epochs\" and set them to 1 to reduce testing time."
    echo ""
    echo "Code style check options:"
    echo "    --no-run          : don't run notebooks"
    echo "    --no-checks       : don't run code checks"
    echo "    --autofix         : autofix where possible"
    echo "    --cell-standard   : check guidelines standards such as ## setup environment cell blocks"
    echo "    --copyright       : check whether every source code and notebook has a copyright header"
    echo "    -f, --failfast    : stop on first error"
    echo "    -p, --pattern     : pattern of files to be run (added to \`find . -type f -name *.ipynb -and ! -wholename *.ipynb_checkpoints*\`)"
    echo "    -h, --help        : show this help message and exit"
    echo "    -t, --test        : shortcut to run a single notebook using pattern \`-and -wholename\`"
    echo "    -v, --version     : show MONAI and system version information and exit"
    echo "    --verbose         : show papermill logs when testing the noteboobks"
    echo ""
    echo "Examples:"
    echo "./runner.sh                             # run full tests (${green}recommended before making pull requests${noColor})."
    echo "./runner.sh --no-run                    # don't run the notebooks."
    echo "./runner.sh --no-checks                 # don't run code checks."
    echo "./runner.sh -t 2d_classification/mednist_tutorial.ipynb"
    echo "                                        # test if notebook mednist_tutorial.ipynb runs properly in test."
    echo "./runner.sh --pattern \"-and \( -name '*read*' -or -name '*load*' \) -and ! -wholename '*acceleration*'\""
    echo "                                        # check filenames containing \"read\" or \"load\", but not if the"
    echo "                                          whole path contains \"deepgrow\"."
    echo "./runner.sh --kernelspec \"kernel\"       # Set the kernelspec value used to run notebooks, default is \"python3\"."
    echo "./runner.sh --no-checks --no-run --copyright
    echo "                                        # test if all notebooks and scripts have the copyright header"
    echo "./runner.sh --no-checks --no-run --cell-standard
    echo "                                        # test if all notebooks follow the cell standards in contributing guidelines"
    echo ""
    echo "${separator}For bug reports, questions, and discussions, please file an issue at:"
    echo "    https://github.com/Project-MONAI/MONAI/issues/new/choose"
    echo ""
    echo "To choose an alternative python executable, set the environmental variable, \"MONAI_PY_EXE\"."
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
        --verbose)
            papermill_opt=" --log-output --log-level DEBUG "
        ;;
        --no-checks)
            doChecks=false
        ;;
        --autofix)
            autofix=true
        ;;
        --cell-standard)
            doStandardizeCells=true
        ;;
        --copyright)
            doCopyright=true
        ;;
        -f|--failfast)
            failfast=true
        ;;
        -p|--pattern)
            pattern+="$2"
            echo $pattern
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
        -t|--test)
            pattern+="-and -wholename ./$2"
            echo $pattern
            shift
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
    success=$?
    set -e
    if [ ${success} -ne 0 ]; then
        print_error_msg "Missing package: $1 (trying pip install -r requirements.txt)"
        ${PY_EXE} -m pip install -r requirements.txt
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

if [ $doCopyright = true ]
then
    check_installed nbformat
    # check copyright headers
    copyright_bad=0
    copyright_all=0
    license="http://www.apache.org/licenses/LICENSE-2.0"
    while read -r fname; do
        copyright_all=$((copyright_all + 1))
        if ! grep "$license" "$fname" > /dev/null; then
            print_error_msg "Missing the license header in file: $fname"
            copyright_bad=$((copyright_bad + 1))
        fi
    done <<< "$(find "$(pwd)" -type f \
        -and -name "*.py" -or -name "*.sh" -or -name "*.cpp" -or -name "*.cu" -or -name "*.h")"

    while read -r fname; do
        copyright_all=$((copyright_all + 1))
        if [[ $(${NB_TEST} verify -f "$fname" -i 0 -k "$license") != true ]]; then
            print_error_msg "Missing the license header the first markdown cell of file: $fname"
            copyright_bad=$((copyright_bad + 1))
        fi
    done <<< "$(find "$(pwd)" -type f -name "*.ipynb" -and ! -wholename "*.ipynb_checkpoints*")"

    if [[ ${copyright_bad} -eq 0 ]];
    then
        echo "${green}Source code copyright headers checked ($copyright_all).${noColor}"
    else
        echo "Please add the licensing header to the file ($copyright_bad of $copyright_all files)."
        echo "  See also: https://github.com/Project-MONAI/tutorials/blob/main/CONTRIBUTING.md#add-license"
        echo ""
        exit 1
    fi
fi

if [ $doStandardizeCells = true ]
then
    # check guideline requirements on standard cells
    standards_all=0
    standards_bad=0
    while read -r fname; do
        standards_all=$((standards_all + 1))
        standardized=true

        code_ind=0
        IFS=' ' read -r -a code_cell_counts <<< $(${NB_TEST} count -f "$fname" --type code)

        for element in ${code_cell_counts[@]} ; do
            if [[ $element != 0 ]]; then
                break
            fi
            code_ind=$((code_ind + 1))
        done

        # there should be at least one code cell
        if [[ $code_ind == ${#code_cell_counts[@]} ]]; then
            print_error_msg "Missing code cells in the file: $fname"
            standardized=false
        fi

        if [[ $code_ind == 0 ]]; then
            print_error_msg "Missing necessary markdown (e.g. copyright header) in the file: $fname"
            print_error_msg "Missing the \"Setup environment\" before the first code cell of file: $fname"
            standardized=false
        else
            # the second cell should be in markdown and contain "Setup environment"
            if [[ $(${NB_TEST} verify -f "$fname" -i $((code_ind - 1)) -k "Setup [eE]nvironment") != true ]]; then
                print_error_msg "\"Setup environment\" is missing or not right before the first code cell of file: $fname"
                standardized=false
            fi
        fi

        # the third cell should be code and contain "pip install"
        if [[ $(${NB_TEST} verify -f "$fname" -i $code_ind -k "pip install" --type code) != true ]]; then
            print_error_msg "Missing the shell command \"pip install -q\" in the first code cell of file: $fname"
            standardized=false
        fi

        # if import is used, then it should have the Setup import(s) markdown
        if [[ $(${NB_TEST} verify -f "$fname" -k "(^import|[\n\r]import|^from|[\n\r]from)" --type code) == true ]]
        then
            if [[ $(${NB_TEST} verify -f "$fname" -i $((code_ind + 1)) -k "Setup import") != true ]]; then
                print_error_msg "Missing the \"Setup imports\" after the first code cell of file: $fname"
                standardized=false
            fi

            if [[ $(${NB_TEST} verify -f "$fname" -i $((code_ind + 2)) -k "print_config()" --type code) != true ]]; then
                print_error_msg "print_config() cannot be found after the \"Setup imports\" markdown cell in file: $fname"
                standardized=false
            fi
        fi

        # the number of lines in text outputs should be under the limit
        # other outputs such as html data won't be counted
        ind=0
        IFS=' ' read -r -a text_lines <<< $(${NB_TEST} count -f "$fname" -k "\n" --type code --field outputs -n text)
        for element in ${text_lines[@]}; do
            if [[ $element -ge $NB_OUTPUT_LINE_CAP ]]; then
                standardized=false
                print_error_msg "Output text in cell #$ind has more than $NB_OUTPUT_LINE_CAP lines in file: $fname"
                break
            fi
            ind=$((ind + 1))
        done

        if [[ $standardized == false ]]; then
            standards_bad=$((standards_bad + 1))
        fi
    done <<< "$(find "$(pwd)" -type f -name "*.ipynb" -and ! -wholename "*.ipynb_checkpoints*")"

    if [[ ${standards_bad} -eq 0 ]]; then
        echo "${green}Notebook check ($standards_all).${noColor}"
    else
        echo "Please fix the notebook formats in ($standards_bad of $standards_all files)."
        echo "  See also: https://github.com/Project-MONAI/tutorials/blob/main/CONTRIBUTING.md#create-a-notebook"
        echo ""
        exit 1
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

# Get notebooks (pattern is an empty string unless the user specifies otherwise)
files=($(echo $pattern | xargs find . -type f -name "*.ipynb" -and ! -wholename "*.ipynb_checkpoints*"))
if [[ $files == "" ]]; then
    print_error_msg "No files match pattern"
    exit 0
fi
echo "Notebook files to be tested:"
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
                --pipe "autopep8 - --ignore W291,E203 --max-line-length 120" \
                --pipe "sed 's/ = list()/ = []/'"
        fi

        # to check flake8, convert to python script, don't check
        # magic cells, and don't check line length for comment
        # lines (as this includes markdown), and then run flake8
        echo Checking PEP8 compliance...
        jupytext "$filename" --opt custom_cell_magics="writefile" -w --to script -o - | \
            sed 's/\(^\s*\)%/\1pass  # %/' | \
            sed 's/\(^#.*\)$/\1  # noqa: E501/' | \
            flake8 - --show-source --extend-ignore=E203 --max-line-length 120
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

        skipRun=false

        for skip_pattern in "${skip_run_papermill[@]}"; do
            if [[  $file =~ $skip_pattern ]]; then
                echo "Skip Pattern Match"
                skipRun=true
                break
            fi
        done

        if [ $skipRun = true ]; then
            echo "Skipping"
            continue
        fi

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

        # FIXME: https://github.com/Project-MONAI/MONAI/issues/4354
        protobuf_major_version=$(${PY_EXE} -m pip list | grep '^protobuf ' | tr -s ' ' | cut -d' ' -f2 | cut -d'.' -f1)
        if [ "$protobuf_major_version" -ge "4" ]
        then
            export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
        else
            unset PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION
        fi

        cmd=$(echo "papermill ${papermill_opt} --progress-bar -k ${kernelspec}")
        echo "$cmd"
        time out=$(echo "$notebook" | eval "$cmd")
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
