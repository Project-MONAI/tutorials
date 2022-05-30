# Copyright 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import json
import os
import random

import numpy as np
import pandas as pd
from preprocess_dicom import dicom_preprocess
from sklearn.model_selection import GroupKFold

# density labels
# 1 - fatty
# 2 - scattered fibroglandular density
# 3 - heterogeneously dense
# 4 - extremely dense


def preprocess(dicom_root, out_path, ids, images, densities, process_image=True):
    data_list = []
    dc_tags = []
    saved_filenames = []
    assert len(ids) == len(images) == len(densities)
    for i, (id, image, density) in enumerate(zip(ids, images, densities)):
        if (i + 1) % 200 == 0:
            print(f"processing {i+1} of {len(ids)}...")
        dir_name = image.split(os.path.sep)[0]
        img_file = glob.glob(
            os.path.join(dicom_root, dir_name, "**", "*.dcm"), recursive=True
        )
        assert len(img_file) == 1, f"No unique dicom image found for {dir_name}!"
        save_prefix = os.path.join(out_path, dir_name)
        if process_image:
            _success, _dc_tags = dicom_preprocess(img_file[0], save_prefix)
        else:
            if os.path.isfile(save_prefix + ".npy"):
                _success = True
            else:
                _success = False
            _dc_tags = []
        if _success and density >= 1:  # label can be 0 sometimes, excluding those cases
            dc_tags.append(_dc_tags)
            data_list.append(
                {
                    "patient_id": id,
                    "image": dir_name + ".npy",
                    "label": int(density - 1),
                }
            )
            saved_filenames.append(dir_name + ".npy")
    return data_list, dc_tags, saved_filenames


def write_datalist(save_datalist_file, data_set):
    os.makedirs(os.path.dirname(save_datalist_file), exist_ok=True)
    with open(save_datalist_file, "w") as f:
        json.dump(data_set, f, indent=4)
    print(f"Data list saved at {save_datalist_file}")


def get_indices(all_ids, search_ids):
    indices = []
    for _id in search_ids:
        _indices = np.where(all_ids == _id)
        indices.extend(_indices[0].tolist())
    return indices


def main():
    process_image = True  # set False if dicoms have already been preprocessed

    out_path = "./data/preprocessed"  # YOUR DEST FOLDER SHOULD BE WRITTEN HERE
    out_dataset_prefix = "./data/dataset"

    # Input folders
    label_root = "/media/hroth/Elements/NVIDIA/Data/CBIS-DDSM/"
    dicom_root = "/media/hroth/Elements/NVIDIA/Data/CBIS-DDSM/DICOM/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM"
    n_clients = 3

    """ Run preprocessing """

    """ 1. Load the label data """
    random.seed(0)

    label_files = [
        os.path.join(label_root, "mass_case_description_train_set.csv"),
        os.path.join(label_root, "calc_case_description_train_set.csv"),
        os.path.join(label_root, "mass_case_description_test_set.csv"),
        os.path.join(label_root, "calc_case_description_test_set.csv"),
    ]

    breast_densities = []
    patients_ids = []
    image_file_path = []

    # read annotations
    for label_file in label_files:
        print(f"add {label_file}")
        label_data = pd.read_csv(label_file)
        unique_images, unique_indices = np.unique(
            label_data["image file path"], return_index=True
        )
        print(
            f"including {len(unique_images)} unique images of {len(label_data['image file path'])} image entries"
        )

        try:
            breast_densities.extend(label_data["breast_density"][unique_indices])
        except BaseException:
            breast_densities.extend(label_data["breast density"][unique_indices])
        patients_ids.extend(label_data["patient_id"][unique_indices])
        image_file_path.extend(label_data["image file path"][unique_indices])

    assert len(breast_densities) == len(patients_ids) == len(image_file_path), (
        f"Mismatch between label data, breast_densities: "
        f"{len(breast_densities)}, patients_ids: {len(patients_ids)}, image_file_path: {len(image_file_path)}"
    )
    print(f"Read {len(image_file_path)} data entries.")

    """ 2. Split the data """

    # shuffle data
    label_data = list(zip(breast_densities, patients_ids, image_file_path))
    random.shuffle(label_data)
    breast_densities, patients_ids, image_file_path = zip(*label_data)

    # Split data
    breast_densities = np.array(breast_densities)
    patients_ids = np.array(patients_ids)
    image_file_path = np.array(image_file_path)

    unique_patient_ids = np.unique(patients_ids)
    n_patients = len(unique_patient_ids)
    print(f"Found {n_patients} patients.")

    # generate splits using roughly the same ratios as for challenge data:
    n_train_challenge = 60_000
    n_val_challenge = 6_500
    n_test_challenge = 40_000
    test_ratio = n_test_challenge / (
        n_train_challenge + n_val_challenge + n_test_challenge
    )
    val_ratio = n_val_challenge / (
        n_val_challenge + n_test_challenge
    )  # test cases will be removed at this point

    # use groups to avoid patient overlaps
    # test split
    n_splits = int(np.ceil(len(image_file_path) / (len(image_file_path) * test_ratio)))
    print(
        f"Splitting into {n_splits} folds for test split. (Only the first fold is used.)"
    )
    group_kfold = GroupKFold(n_splits=n_splits)
    for train_val_index, test_index in group_kfold.split(
        image_file_path, breast_densities, groups=patients_ids
    ):
        break  # just use first fold
    test_images = image_file_path[test_index]
    test_patients_ids = patients_ids[test_index]
    test_densities = breast_densities[test_index]

    # train/val splits
    train_val_images = image_file_path[train_val_index]
    train_val_patients_ids = patients_ids[train_val_index]
    train_val_densities = breast_densities[train_val_index]

    n_splits = int(np.ceil(len(image_file_path) / (len(image_file_path) * val_ratio)))
    print(
        f"Splitting into {n_splits} folds for train/val splits. (Only the first fold is used.)"
    )
    group_kfold = GroupKFold(n_splits=n_splits)
    for train_index, val_index in group_kfold.split(
        train_val_images, train_val_densities, groups=train_val_patients_ids
    ):
        break  # just use first fold

    train_images = train_val_images[train_index]
    train_patients_ids = train_val_patients_ids[train_index]
    train_densities = train_val_densities[train_index]

    val_images = train_val_images[val_index]
    val_patients_ids = train_val_patients_ids[val_index]
    val_densities = train_val_densities[val_index]

    # check that there is no patient overlap
    assert (
        len(np.intersect1d(train_patients_ids, val_patients_ids)) == 0
    ), "Overlapping patients in train and validation!"
    assert (
        len(np.intersect1d(train_patients_ids, test_patients_ids)) == 0
    ), "Overlapping patients in train and test!"
    assert (
        len(np.intersect1d(val_patients_ids, test_patients_ids)) == 0
    ), "Overlapping patients in validation and test!"

    n_total = len(train_images) + len(val_images) + len(test_images)
    print(20 * "-")
    print(f"Train : {len(train_images)} ({100*len(train_images)/n_total:.2f}%)")
    print(f"Val   : {len(val_images)}   ({100*len(val_images)/n_total:.2f}%)")
    print(f"Test  : {len(test_images)}  ({100*len(test_images)/n_total:.2f}%)")
    print(20 * "-")
    print(f"Total : {n_total}")
    assert n_total == len(image_file_path), (
        f"mismatch between total split images ({n_total})"
        f" and length of all images {len(image_file_path)}!"
    )

    """ split train/validation dataset for n_clients """
    # Split and avoid patient overlap
    unique_train_patients_ids = np.unique(train_patients_ids)
    split_train_patients_ids = np.array_split(unique_train_patients_ids, n_clients)

    unique_val_patients_ids = np.unique(val_patients_ids)
    split_val_patients_ids = np.array_split(unique_val_patients_ids, n_clients)

    unique_test_patients_ids = np.unique(test_patients_ids)
    split_test_patients_ids = np.array_split(unique_test_patients_ids, n_clients)

    """ 3. Preprocess the images """
    dc_tags = []
    saved_filenames = []
    for c in range(n_clients):
        site_name = f"site-{c+1}"
        print(f"Preprocessing training set of client {site_name}")
        _curr_patient_ids = split_train_patients_ids[c]
        _curr_indices = get_indices(train_patients_ids, _curr_patient_ids)
        train_list, _dc_tags, _saved_filenames = preprocess(
            dicom_root,
            out_path,
            train_patients_ids[_curr_indices],
            train_images[_curr_indices],
            train_densities[_curr_indices],
            process_image=process_image,
        )
        print(
            f"Converted {len(train_list)} of {len(train_patients_ids)} training images"
        )
        dc_tags.extend(_dc_tags)
        saved_filenames.extend(_saved_filenames)

        print("Preprocessing validation")
        _curr_patient_ids = split_val_patients_ids[c]
        _curr_indices = get_indices(val_patients_ids, _curr_patient_ids)
        val_list, _dc_tags, _saved_filenames = preprocess(
            dicom_root,
            out_path,
            val_patients_ids[_curr_indices],
            val_images[_curr_indices],
            val_densities[_curr_indices],
            process_image=process_image,
        )
        print(f"Converted {len(val_list)} of {len(val_patients_ids)} validation images")
        dc_tags.extend(_dc_tags)
        saved_filenames.extend(_saved_filenames)

        print("Preprocessing testing")
        _curr_patient_ids = split_test_patients_ids[c]
        _curr_indices = get_indices(test_patients_ids, _curr_patient_ids)
        test_list, _dc_tags, _saved_filenames = preprocess(
            dicom_root,
            out_path,
            test_patients_ids[_curr_indices],
            test_images[_curr_indices],
            test_densities[_curr_indices],
            process_image=process_image,
        )
        print(f"Converted {len(test_list)} of {len(test_patients_ids)} testing images")
        dc_tags.extend(_dc_tags)
        saved_filenames.extend(_saved_filenames)

        data_set = {
            "train": train_list,  # will stay the same for both phases
            "test1": val_list,  # like phase 1 leaderboard
            "test2": test_list,  # like phase 2 - final leaderboard
        }
        write_datalist(f"{out_dataset_prefix}_{site_name}.json", data_set)

    print(50 * "=")
    print(
        f"Successfully converted a total {len(saved_filenames)} of {len(image_file_path)} images."
    )

    # check that there were no duplicated files
    assert len(saved_filenames) == len(
        np.unique(saved_filenames)
    ), f"Not all generated files ({len(saved_filenames)}) are unique ({len(np.unique(saved_filenames))})!"

    print(f"Data lists saved wit prefix {out_dataset_prefix}")
    print(50 * "=")
    print("Processed unique DICOM tags", np.unique(dc_tags))


if __name__ == "__main__":
    main()
