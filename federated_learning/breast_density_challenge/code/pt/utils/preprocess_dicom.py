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

import os

import cv2
import numpy as np
import pydicom
import skimage.io


def dicom_preprocess(dicom_file, save_prefix):
    try:
        # Read needed dicom tags
        ds = pydicom.dcmread(dicom_file)  # , stop_before_pixels=True)
        try:
            code = ds.ViewCodeSequence[0].ViewModifierCodeSequence[0].CodeMeaning
        except BaseException:
            code = None

        # Filter image
        dc_tags = f"BS={ds.BitsStored};PI={ds.PhotometricInterpretation};Modality={ds.Modality};PatientOrientation={ds.PatientOrientation};Code={code}"
        if ds.PatientOrientation == "MLO" or ds.PatientOrientation == "CC":
            curr_img = ds.pixel_array
            curr_img = np.squeeze(curr_img).T.astype(np.float)

            # Can be modified as well to handle other bit and monochrome combinations
            if (ds.BitsStored == 16) and "2" in ds.PhotometricInterpretation:
                curr_img = curr_img / 65535.0
            else:
                raise ValueError(dicom_file + " - unsupported dicom tags: " + dc_tags)

            # Resize and replicate into 3 channels
            curr_img = cv2.resize(curr_img, (224, 224))
            curr_img = np.concatenate(
                (
                    curr_img[:, :, np.newaxis],
                    curr_img[:, :, np.newaxis],
                    curr_img[:, :, np.newaxis],
                ),
                axis=-1,
            )
            # Save output file
            assert curr_img.min() >= 0 and curr_img.max() <= 1.0

            os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
            np.save(save_prefix + ".npy", curr_img.astype(np.float32))
            skimage.io.imsave(
                save_prefix + ".png", (255 * curr_img / curr_img.max()).astype(np.uint8)
            )
        else:
            raise ValueError(
                "Error: " + dicom_file + " - not a valid image file: " + dc_tags
            )
    except BaseException as e:
        print(f"[WARNING] Reading {dicom_file} failed with Exception: {e}")
        return False, f"{dicom_file} failed"

    return True, dc_tags
