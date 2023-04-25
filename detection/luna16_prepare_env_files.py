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

import os
import json
import logging
import sys


def main():
    #  ------------- Modification starts -------------
    raw_data_base_dir = "/orig_datasets/"  # the directory of the raw images
    resampled_data_base_dir = "/datasets/"  # the directory of the resampled images
    downloaded_datasplit_dir = "LUNA16_datasplit"  # the directory of downloaded data split files

    out_trained_models_dir = "trained_models"  # the directory to save trained model weights
    out_tensorboard_events_dir = "tfevent_train"  # the directory to save tensorboard training curves
    out_inference_result_dir = "result"  # the directory to save predicted boxes for inference

    # if deal with mhd/raw data, set it to be None
    dicom_meta_data_csv = None
    # if deal with DICOM data, also need metadata.csv
    # dicom_meta_data_csv = "/orig_datasets/dicom/metadata.csv"
    #  ------------- Modification ends ---------------

    try:
        os.mkdir(out_trained_models_dir)
    except FileExistsError:
        pass

    try:
        os.mkdir(out_tensorboard_events_dir)
    except FileExistsError:
        pass

    try:
        os.mkdir(out_inference_result_dir)
    except FileExistsError:
        pass

    # generate env json file for image resampling
    out_file = "config/environment_luna16_prepare.json"
    env_dict = {}
    env_dict["orig_data_base_dir"] = raw_data_base_dir
    env_dict["data_base_dir"] = resampled_data_base_dir
    if dicom_meta_data_csv != None:
        env_dict["data_list_file_path"] = os.path.join(downloaded_datasplit_dir, "dicom_original/dataset_fold0.json")
    else:
        env_dict["data_list_file_path"] = os.path.join(downloaded_datasplit_dir, "mhd_original/dataset_fold0.json")
    if dicom_meta_data_csv != None:
        env_dict["dicom_meta_data_csv"] = dicom_meta_data_csv
    with open(out_file, "w") as outfile:
        json.dump(env_dict, outfile, indent=4)

    # generate env json file for training and inference
    for fold in range(10):
        out_file = "config/environment_luna16_fold" + str(fold) + ".json"
        env_dict = {}
        env_dict["model_path"] = os.path.join(out_trained_models_dir, "model_luna16_fold" + str(fold) + ".pt")
        env_dict["data_base_dir"] = resampled_data_base_dir
        env_dict["data_list_file_path"] = os.path.join(downloaded_datasplit_dir, "dataset_fold" + str(fold) + ".json")
        env_dict["tfevent_path"] = os.path.join(out_tensorboard_events_dir, "luna16_fold" + str(fold))
        env_dict["result_list_file_path"] = os.path.join(
            out_inference_result_dir, "result_luna16_fold" + str(fold) + ".json"
        )
        with open(out_file, "w") as outfile:
            json.dump(env_dict, outfile, indent=4)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
