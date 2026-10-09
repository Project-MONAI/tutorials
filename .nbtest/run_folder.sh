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

# Run every notebook under a folder (recursively), skipping those on the project
# skip list, and print one NBTEST_RESULT json line per notebook.
#
# Usage: run_folder.sh <folder> [timeout_seconds]
set -u
cd "$(dirname "$0")/.." || exit 1
export MONAI_DATA_DIRECTORY="$(pwd)/.nbtest/data"
PY=.venv/bin/python
FOLDER="$1"
TIMEOUT="${2:-1800}"

# patterns (basename substrings) that the project runner skips with papermill
SKIP_PATTERNS=(
  05_spleen_segmentation_lightning GDS_dataset MRI_reconstruction
  TCIA_PROSTATEx_Prostate_MRI_Anatomy_Model TensorRT_inference_acceleration
  TorchIO_MONAI_PyTorch_Lightning active_learning benchmark_global_mutual_information
  federated_learning finetune_vista3d_for_hugging_face_pipeline full_gpu_inference_pipeline
  generate_random_permutations hovernet_torch idc_dataset image_restoration
  learn2reg_nlst_paired_lung_ct learn2reg_oasis_unpaired_brain_mr maisi_inference_tutorial
  mednist_classifier_ray monailabel_ nuclei_classification nuclick
  preprocess_detect_scene_and_split_fold preprocess_extract_images_from_video
  preprocess_to_build_detection_dataset profiling_train_base_nvtx
  spleen_segmentation_3d_visualization_basic ssl_finetune ssl_train swinunetr_finetune
  tcia_dataset transchex_openi transfer_mmar transform_visualization
  transforms_update_meta_data unet_segmentation_3d_ignite_clearml unetr_
  video_seg vista_2d_tutorial_monai
)

is_skipped() {
  local f="$1"
  for p in "${SKIP_PATTERNS[@]}"; do
    [[ "$f" == *"$p"* ]] && return 0
  done
  return 1
}

mapfile -t files < <(find "$FOLDER" -name "*.ipynb" -not -path "*checkpoint*" -not -name ".nbtest_in_*" -not -name "*.out.ipynb" | sort)
for nb in "${files[@]}"; do
  if is_skipped "$nb"; then
    echo "NBTEST_RESULT {\"notebook\": \"$nb\", \"status\": \"skipped\", \"elapsed_s\": 0, \"reason\": \"project skip list\"}"
    continue
  fi
  slug=$(echo "$nb" | tr '/.' '--')
  log=".nbtest/logs/${slug}.log"
  $PY .nbtest/run_nb.py "$nb" --timeout "$TIMEOUT" --kernel monai-venv > "$log" 2>&1
  grep NBTEST_RESULT "$log" || echo "NBTEST_RESULT {\"notebook\": \"$nb\", \"status\": \"error\", \"reason\": \"no result line; see $log\"}"
done
