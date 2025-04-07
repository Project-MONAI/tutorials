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

import numpy as np


def get_masked_data(label_data, image_data, labels):
    """
    Extracts and returns the image data corresponding to specified labels within a 3D volume.

    This function efficiently masks the `image_data` array based on the provided `labels` in the `label_data` array.
    The function handles cases with both a large and small number of labels, optimizing performance accordingly.

    Args:
        label_data (np.ndarray): A NumPy array containing label data, representing different anatomical
                                 regions or classes in a 3D medical image.
        image_data (np.ndarray): A NumPy array containing the image data from which the relevant regions
                                 will be extracted.
        labels (list of int): A list of integers representing the label values to be used for masking.

    Returns:
        np.ndarray: A NumPy array containing the elements of `image_data` that correspond to the specified
                    labels in `label_data`. If no labels are provided, an empty array is returned.

    Raises:
        ValueError: If `image_data` and `label_data` do not have the same shape.

    Example:
        label_int_dict = {"liver": [1], "kidney": [5, 14]}
        masked_data = get_masked_data(label_data, image_data, label_int_dict["kidney"])
    """

    # Check if the shapes of image_data and label_data match
    if image_data.shape != label_data.shape:
        raise ValueError(
            f"Shape mismatch: image_data has shape {image_data.shape}, "
            f"but label_data has shape {label_data.shape}. They must be the same."
        )

    if not labels:
        return np.array([])  # Return an empty array if no labels are provided

    labels = list(set(labels))  # remove duplicate items

    # Optimize performance based on the number of labels
    num_label_acceleration_thresh = 3
    if len(labels) >= num_label_acceleration_thresh:
        # if many labels, np.isin is faster
        mask = np.isin(label_data, labels)
    else:
        # Use logical OR to combine masks if the number of labels is small
        mask = np.zeros_like(label_data, dtype=bool)
        for label in labels:
            mask = np.logical_or(mask, label_data == label)

    # Retrieve the masked data
    masked_data = image_data[mask.astype(bool)]

    return masked_data


def is_outlier(statistics, image_data, label_data, label_int_dict):
    """
    Perform a quality check on the generated image by comparing its statistics with precomputed thresholds.

    Args:
        statistics (dict): Dictionary containing precomputed statistics including mean +/- 3sigma ranges.
        image_data (np.ndarray): The image data to be checked, typically a 3D NumPy array.
        label_data (np.ndarray): The label data corresponding to the image, used for masking regions of interest.
        label_int_dict (dict): Dictionary mapping label names to their corresponding integer lists.
            e.g., label_int_dict = {"liver": [1], "kidney": [5, 14]}

    Returns:
        dict: A dictionary with labels as keys, each containing the quality check result,
              including whether it's an outlier, the median value, and the thresholds used.
              If no data is found for a label, the median value will be `None` and `is_outlier` will be `False`.

    Example:
        # Example input data
        statistics = {
            "liver": {
                "sigma_6_low": -21.596463547885904,
                "sigma_6_high": 156.27881534763367
            },
            "kidney": {
                "sigma_6_low": -15.0,
                "sigma_6_high": 120.0
            }
        }
        label_int_dict = {
            "liver": [1],
            "kidney": [5, 14]
        }
        image_data = np.random.rand(100, 100, 100)  # Replace with actual image data
        label_data = np.zeros((100, 100, 100))  # Replace with actual label data
        label_data[40:60, 40:60, 40:60] = 1  # Example region for liver
        label_data[70:90, 70:90, 70:90] = 5  # Example region for kidney
        result = is_outlier(statistics, image_data, label_data, label_int_dict)
    """
    outlier_results = {}

    for label_name, stats in statistics.items():
        # Get the thresholds from the statistics
        low_thresh = min(stats["sigma_6_low"], stats["percentile_0_5"])  # or "sigma_12_low" depending on your needs
        high_thresh = max(stats["sigma_6_high"], stats["percentile_99_5"])  # or "sigma_12_high" depending on your needs

        if label_name == "bone":
            high_thresh = 1000.0

        # Retrieve the corresponding label integers
        labels = label_int_dict.get(label_name, [])
        masked_data = get_masked_data(label_data, image_data, labels)
        masked_data = masked_data[~np.isnan(masked_data)]

        if len(masked_data) == 0 or masked_data.size == 0:
            outlier_results[label_name] = {
                "is_outlier": False,
                "median_value": None,
                "low_thresh": low_thresh,
                "high_thresh": high_thresh,
            }
            continue

        # Compute the median of the masked region
        median_value = np.nanmedian(masked_data)

        if np.isnan(median_value):
            median_value = None
            is_outlier = False
        else:
            # Determine if the median value is an outlier
            is_outlier = median_value < low_thresh or median_value > high_thresh

        outlier_results[label_name] = {
            "is_outlier": is_outlier,
            "median_value": median_value,
            "low_thresh": low_thresh,
            "high_thresh": high_thresh,
        }

    return outlier_results
