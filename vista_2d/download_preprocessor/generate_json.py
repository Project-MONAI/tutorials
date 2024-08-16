import argparse
import gc
import json
import os
import shutil
import time
import warnings
import zipfile

import imageio.v3 as imageio
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from sklearn.model_selection import KFold

# from skimage.io import imsave
# from skimage.measure import label
# import imageio


def min_label_precision(label):
    lm = label.max()

    if lm <= 255:
        label = label.astype(np.uint8)
    elif lm <= 65535:
        label = label.astype(np.uint16)
    else:
        label = label.astype(np.uint32)

    return label


def guess_convert_to_uint16(img, margin=30):
    """
    Guess a multiplier that makes all pixels integers.
    The input img (each channel) is already in the range 0..1, they must have been converted from uint16 integers as image / scale,
    where scale was the unknown max intensity. We could guess the scale by looking at unique values: 1/np.min(np.diff(np.unique(im)).
    the hypothesis is that it will be more accurate recovery of the original image, instead of doing a simple (img*65535).astype(np.uint16)
    """

    for i in range(img.shape[0]):
        im = img[i]

        if im.any():
            start = time.time()
            imsmall = im[::4, ::4]  # subsample
            # imsmall = im

            scale = int(np.round(1 / np.min(np.diff(np.unique(imsmall)))))  # guessing scale
            test = [
                (np.sum((imsmall * k) % 1)) for k in range(scale - margin, scale + margin)
            ]  # finetune, guess a multiplier that makes all pixels integers
            sid = np.argmin(test)  # fine tune scale
            # print('guessing scale', scale, test[margin], 'fine tuning scale', scale - margin + sid, 'dif', test[sid], 'time', time.time()-start)

            if scale < 16000 or scale > 16400:
                # print(imsmall.shape)
                # print(np.unique(imsmall))
                # print(np.diff(np.unique(imsmall)))
                # print(np.min(np.diff(np.unique(imsmall))))
                warnings.warn("scale not in expected range")
                print(
                    "guessing scale",
                    scale,
                    test[margin],
                    "fine tuning scale",
                    scale - margin + sid,
                    "dif",
                    test[sid],
                    "time",
                    time.time() - start,
                )

                scale = 16384
            else:
                scale = scale - margin + sid

            scale = min(
                65535, scale * 4
            )  # all the recovered scale values seems to be up to 16384, we can stretch to 65535 (for better visualization, most tiff viewers expect that range)
            img[i] = im * scale

    img = img.astype(np.uint16)
    return img


def concatenate_masks(mask_dir):
    labeled_mask = None
    i = 0
    for filename in sorted(os.listdir(mask_dir)):
        if filename.endswith(".png"):
            mask = imageio.imread(os.path.join(mask_dir, filename))
            if labeled_mask is None:
                labeled_mask = np.zeros(shape=mask.shape, dtype=np.uint16)
            labeled_mask[mask > 0] = i
            i = i + 1

    if i <= 255:
        labeled_mask = labeled_mask.astype(np.uint8)

    return labeled_mask


# def concatenate_masks(mask_dir):
#     masks = []
#     for filename in sorted(os.listdir(mask_dir)):
#         if filename.endswith('.png'):
#             mask = imageio.imread(os.path.join(mask_dir, filename))
#             masks.append(mask)
#     concatenated_mask = np.any(masks, axis=0).astype(np.uint8)
#     labeled_mask = label(concatenated_mask)
#     return labeled_mask

# def normalize_image(image):
#     # Convert to float and normalize each channel
#     image = image.astype(np.float32)
#     for i in range(3):
#         channel = image[..., i]
#         channel_min = np.min(channel)
#         channel_max = np.max(channel)
#         if channel_max - channel_min != 0:
#             image[..., i] = (channel - channel_min) / (channel_max - channel_min)
#     return image


def get_filenames_exclude_masks(dir1, target_string):
    filenames = []
    # Combine lists of files from both directories
    files = os.listdir(dir1)
    # Filter files that contain the target string but exclude 'masks'
    filenames = [f for f in files if target_string in f and "masks" not in f]

    return filenames


def remove_overlaps(masks, medians, overlap_threshold=0.75):
    """replace overlapping mask pixels with mask id of closest mask
    if mask fully within another mask, remove it
    masks = Nmasks x Ly x Lx
    """
    cellpix = masks.sum(axis=0)
    igood = np.ones(masks.shape[0], "bool")
    for i in masks.sum(axis=(1, 2)).argsort():
        npix = float(masks[i].sum())
        noverlap = float(masks[i][cellpix > 1].sum())
        if noverlap / npix >= overlap_threshold:
            igood[i] = False
            cellpix[masks[i] > 0] -= 1
            # print(cellpix.min())
    print(f"removing {(~igood).sum()} masks")
    masks = masks[igood]
    medians = medians[igood]
    cellpix = masks.sum(axis=0)
    overlaps = np.array(np.nonzero(cellpix > 1.0)).T
    dists = ((overlaps[:, :, np.newaxis] - medians.T) ** 2).sum(axis=1)
    tocell = np.argmin(dists, axis=1)
    masks[:, overlaps[:, 0], overlaps[:, 1]] = 0
    masks[tocell, overlaps[:, 0], overlaps[:, 1]] = 1

    # labels should be 1 to mask.shape[0]
    masks = masks.astype(int) * np.arange(1, masks.shape[0] + 1, 1, int)[:, np.newaxis, np.newaxis]
    masks = masks.sum(axis=0)
    gc.collect()
    return masks


def livecell_json_files(dataset_dir, json_f_path):
    """
    This function takes in the directory of livecell extracted dataset as input and
    creates 7 json lists with 5 folds. Separate testing set is recorded in the json list.
    Please note that there are some hard-coded directory names as per the original dataset.
    At the time of creation, the livecell zipfile had 'images' and 'LIVECell_dataset_2021' directories
    """

    # "A172", "BT474", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"
    # TODO "BV2" is being skipped
    cell_type_list = ["A172", "BT474", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]
    for each_cell_tp in cell_type_list:
        for split in ["train", "val", "test"]:
            print(f"Working on split: {split}")

            if split == "test":
                img_path = os.path.join(dataset_dir, "images", "livecell_test_images", each_cell_tp)
                msk_path = os.path.join(dataset_dir, "images", "livecell_test_images", each_cell_tp + "_masks")
            else:
                img_path = os.path.join(dataset_dir, "images", "livecell_train_val_images", each_cell_tp)
                msk_path = os.path.join(dataset_dir, "images", "livecell_train_val_images", each_cell_tp + "_masks")
            if not os.path.exists(msk_path):
                os.makedirs(msk_path)

            # annotation path
            path = os.path.join(
                dataset_dir,
                "livecell-dataset.s3.eu-central-1.amazonaws.com",
                "LIVECell_dataset_2021",
                "annotations",
                "LIVECell_single_cells",
                each_cell_tp.lower(),
                split + ".json",
            )
            annotation = COCO(path)
            # Convert COCO format segmentation to binary mask
            images = annotation.loadImgs(annotation.getImgIds())
            height = []
            width = []
            for index, im in enumerate(images):
                print("Status: {}/{}, Process image: {}".format(index, len(images), im["file_name"]))
                if (
                    im["file_name"] == "BV2_Phase_C4_2_03d00h00m_1.tif"
                    or im["file_name"] == "BV2_Phase_C4_2_03d00h00m_3.tif"
                ):
                    print("Skipping the file: BV2_Phase_C4_2_03d00h00m_1.tif, as it is troublesome")
                    continue
                # load image
                img = Image.open(os.path.join(img_path, im["file_name"])).convert("L")
                height.append(img.size[0])
                width.append(img.size[1])
                # arr = np.asarray(img) #? not used
                # msk = np.zeros(arr.shape)
                # load and display instance annotations
                annIds = annotation.getAnnIds(imgIds=im["id"], iscrowd=None)
                anns = annotation.loadAnns(annIds)
                idx = 1
                medians = []
                masks = []
                k = 0
                for ann in anns:
                    # convert segmentation to binary mask
                    mask = annotation.annToMask(ann)
                    masks.append(mask)
                    ypix, xpix = mask.nonzero()
                    medians.append(np.array([ypix.mean().astype(np.float32), xpix.mean().astype(np.float32)]))
                    k += 1
                    # add instance mask to image mask
                    # msk = np.add(msk, mask*idx)
                    # idx += 1

                masks = np.array(masks).astype(np.int8)
                medians = np.array(medians)
                masks = remove_overlaps(masks, medians, overlap_threshold=0.75)
                gc.collect()

                # ## Create new name for the image and also for the mask and save them as .tif format
                # masks_int32 = masks.astype(np.int32)
                # mask_pil = Image.fromarray(masks_int32, 'I')

                t_filename = im["file_name"]
                # cell_type = t_filename.split('_')[0] #? not used
                new_mask_name = t_filename[:-4] + "_masks.tif"
                # mask_pil.save(os.path.join(msk_path, new_mask_name))
                imageio.imwrite(os.path.join(msk_path, new_mask_name), min_label_precision(masks))
                gc.collect()

            print(f"In total {len(images)} images")

        # The directory containing your files
        # cell_type = 'BV2'
        json_save_path = os.path.join(json_f_path, f"lc_{each_cell_tp}.json")
        directory = os.path.join(dataset_dir, "images", "livecell_train_val_images", each_cell_tp)
        mask_directory = os.path.join(dataset_dir, "images", "livecell_train_val_images", each_cell_tp + "_masks")
        test_directory = os.path.join(dataset_dir, "images", "livecell_test_images", each_cell_tp)
        mask_test_directory = os.path.join(dataset_dir, "images", "livecell_test_images", each_cell_tp + "_masks")
        # List to hold all image-mask pairs
        data_pairs = []
        test_data_pairs = []
        all_data = {}

        # Scan the directory for image files and create pairs
        for filename in os.listdir(directory):
            if filename.endswith(".tif"):
                # Construct the corresponding mask filename
                mask_filename = filename.replace(".tif", "_masks.tif")

                # Check if the corresponding mask file exists
                if os.path.exists(os.path.join(mask_directory, mask_filename)):
                    # Add the pair to the list
                    data_pairs.append(
                        {
                            "image": os.path.join(
                                "livecell_dataset", "images", "livecell_train_val_images", each_cell_tp, filename
                            ),
                            "label": os.path.join(
                                "livecell_dataset",
                                "images",
                                "livecell_train_val_images",
                                f"{each_cell_tp}_masks",
                                mask_filename,
                            ),
                        }
                    )

        # Convert data_pairs to a numpy array for easy indexing by KFold
        data_pairs_array = np.array(data_pairs)

        # Initialize KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Assign fold numbers
        for fold, (train_index, val_index) in enumerate(kf.split(data_pairs_array)):
            for idx in val_index:
                data_pairs_array[idx]["fold"] = fold

        # Convert the array back to a list and sort by fold
        sorted_data_pairs = sorted(data_pairs_array.tolist(), key=lambda x: x["fold"])

        print(sorted_data_pairs)

        # Scan the directory for image files and create pairs
        for filename in os.listdir(test_directory):
            if filename.endswith(".tif"):
                # Construct the corresponding mask filename
                mask_filename = filename.replace(".tif", "_masks.tif")

                # Check if the corresponding mask file exists
                if os.path.exists(os.path.join(mask_test_directory, mask_filename)):
                    # Add the pair to the list
                    test_data_pairs.append(
                        {
                            "image": os.path.join(
                                "livecell_dataset", "images", "livecell_test_images", each_cell_tp, filename
                            ),
                            "label": os.path.join(
                                "livecell_dataset",
                                "images",
                                "livecell_test_images",
                                f"{each_cell_tp}_masks",
                                mask_filename,
                            ),
                        }
                    )

        all_data["training"] = sorted_data_pairs
        all_data["testing"] = test_data_pairs

        with open(json_save_path, "w") as j_file:
            json.dump(all_data, j_file, indent=4)
        j_file.close()


def tissuenet_json_files(dataset_dir, json_f_path):
    """
    This function takes in the directory of TissueNet extracted dataset as input and
    creates 13 json lists with 5 folds each. Separate testing set is recorded in the json list per subset.
    Please note that there are some hard-coded directory names as per the original dataset.
    At the time of creation, the tissuenet 1.0 zipfile had 'train', 'val' and 'test' directories that
    images with paired labels.
    """

    for folder in ["train", "val", "test"]:
        if not os.path.exists(os.path.join(dataset_dir, "tissuenet_1.0", folder)):
            os.mkdir(os.path.join(dataset_dir, "tissuenet_1.0", folder))

    for folder in ["train", "val", "test"]:
        print(f"Working on {folder} directory of tissuenet")
        f_name = f"tissuenet_1.0/tissuenet_v1.0_{folder}.npz"
        dat = np.load(os.path.join(dataset_dir, f_name))
        data = dat["X"]
        labels = dat["y"]
        tissues = dat["tissue_list"]
        platforms = dat["platform_list"]
        tlabels = np.unique(tissues)
        plabels = np.unique(platforms)
        tp = 0
        for t in tlabels:
            for p in plabels:
                ix = ((tissues == t) * (platforms == p)).nonzero()[0]
                tp += 1
                if len(ix) > 0:
                    print(f"Working on {t} {p}")

                    for k, i in enumerate(ix):
                        print(f"Status: {k}/{len(ix)} {tp}/{len(tlabels) * len(plabels)} {t} {p}")
                        img = data[i].transpose(2, 0, 1)
                        label = labels[i][:, :, 0]

                        img = guess_convert_to_uint16(img)  # guess inverse scale and convert to uint16
                        label = min_label_precision(label)

                        if folder == "train":
                            img = img.reshape(2, 2, 256, 2, 256).transpose(0, 1, 3, 2, 4).reshape(2, 4, 256, 256)
                            label = label.reshape(2, 256, 2, 256).transpose(0, 2, 1, 3).reshape(4, 256, 256)

                            zero_channel = np.zeros((1, img.shape[1], img.shape[2], img.shape[3]), dtype=img.dtype)

                            # Concatenate the zero channel with the original array along the first dimension
                            new_array = np.concatenate([img, zero_channel], axis=0)
                            # reshaped_array = np.transpose(new_array, (1, 2, 3, 0))
                            for j in range(4):
                                # imwrite(f'/scratch_2/cell_imaging_2023/tissuenet/tissuenet_v1.0/tissuenet_1_cellpose_way/{folder}/{t}_{p}_{k}_{j}.tif', img[:,j])
                                # imwrite(f'/scratch_2/cell_imaging_2023/tissuenet/tissuenet_v1.0/tissuenet_1_cellpose_way/{folder}/{t}_{p}_{k}_{j}_masks.tif', label[j])
                                img_name = f"{folder}/{t}_{p}_{k}_{j}.tif"
                                mask_name = f"{folder}/{t}_{p}_{k}_{j}_masks.tif"
                                imageio.imwrite(os.path.join(dataset_dir, "tissuenet_1.0", img_name), new_array[:, j])
                                imageio.imwrite(os.path.join(dataset_dir, "tissuenet_1.0", mask_name), label[j])
                        else:
                            zero_channel = np.zeros((1, img.shape[1], img.shape[2]), dtype=img.dtype)
                            new_array = np.concatenate([img, zero_channel], axis=0)
                            # reshaped_array = np.transpose(new_array, (1, 2, 0))
                            img_name = f"{folder}/{t}_{p}_{k}.tif"
                            mask_name = f"{folder}/{t}_{p}_{k}_masks.tif"
                            imageio.imwrite(os.path.join(dataset_dir, "tissuenet_1.0", img_name), new_array)
                            imageio.imwrite(os.path.join(dataset_dir, "tissuenet_1.0", mask_name), label)

    t_p_combos = [
        ["breast", "imc"],
        ["breast", "mibi"],
        ["breast", "vectra"],
        ["gi", "codex"],
        ["gi", "mibi"],
        ["gi", "mxif"],
        ["immune", "cycif"],
        ["immune", "mibi"],
        ["immune", "vectra"],
        ["lung", "cycif"],
        ["lung", "mibi"],
        ["pancreas", "codex"],
        ["pancreas", "vectra"],
        ["skin", "mibi"],
    ]

    for each_t_p in t_p_combos:
        json_f_name = "tn_" + each_t_p[0] + "_" + each_t_p[1] + ".json"
        json_f_subset_path = os.path.join(json_f_path, json_f_name)

        tp_match = each_t_p[0] + "_" + each_t_p[1]
        train_filenames = get_filenames_exclude_masks(os.path.join(dataset_dir, "tissuenet_1.0", "train"), tp_match)
        val_filenames = get_filenames_exclude_masks(os.path.join(dataset_dir, "tissuenet_1.0", "val"), tp_match)
        test_filenames = get_filenames_exclude_masks(os.path.join(dataset_dir, "tissuenet_1.0", "test"), tp_match)

        train_data_list = []
        test_data_list = []

        for each_tf in train_filenames:
            t_dict = {
                "image": os.path.join("tissuenet_dataset", "tissuenet_1.0", "train", each_tf),
                "label": os.path.join("tissuenet_dataset", "tissuenet_1.0", "train", each_tf[:-4] + "_masks.tif"),
            }
            train_data_list.append(t_dict)

        for each_vf in val_filenames:
            t_dict = {
                "image": os.path.join("tissuenet_dataset", "tissuenet_1.0", "val", each_vf),
                "label": os.path.join("tissuenet_dataset", "tissuenet_1.0", "val", each_vf[:-4] + "_masks.tif"),
            }
            train_data_list.append(t_dict)

        for each_tf in test_filenames:
            t_dict = {
                "image": os.path.join("tissuenet_dataset", "tissuenet_1.0", "test", each_tf),
                "label": os.path.join("tissuenet_dataset", "tissuenet_1.0", "test", each_tf[:-4] + "_masks.tif"),
            }
            test_data_list.append(t_dict)

        # print(train_data_list)
        # print(test_data_list)

        # Convert data_pairs to a numpy array for easy indexing by KFold
        data_pairs_array = np.array(train_data_list)

        # Initialize KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Assign fold numbers
        for fold, (train_index, val_index) in enumerate(kf.split(data_pairs_array)):
            for idx in val_index:
                data_pairs_array[idx]["fold"] = fold

        # Convert the array back to a list and sort by fold
        sorted_data_pairs = sorted(data_pairs_array.tolist(), key=lambda x: x["fold"])

        print(sorted_data_pairs)

        all_data = {}
        all_data["training"] = sorted_data_pairs
        all_data["testing"] = test_data_list

        with open(json_f_subset_path, "w") as j_file:
            json.dump(all_data, j_file, indent=4)
        j_file.close()


def omnipose_json_file(dataset_dir, json_path):
    """
    This function takes in the directory of extracted Omnipose dataset as input
    and creates a json list with 5 folds. Please note that only 'bact_phase' and 'bact_fluor' were
    used for creating datasets as they have bacteria the other directiories are worms. Each directory
    has 'train_sorted' and 'test_sorted'.Separate testing set is recorded in the json list.
    Please note that there are some hard-coded directory names as per the original dataset.
    """
    # Define the folders
    op_list = ["bact_fluor", "bact_phase"]
    for each_op in op_list:
        print(f"Working on {each_op} ...")
        images_folder = os.path.join(dataset_dir, each_op, "train_sorted")
        test_images_folder = os.path.join(dataset_dir, each_op, "test_sorted")
        json_f_path = os.path.join(json_path, f"op_{each_op}.json")

        # Initialize the list for training data
        training_data = []

        # Loop through each image file to find its corresponding label file
        sub_dirs = os.listdir(images_folder)
        # Likely Omnipose dataset was created using a Mac and hence the spare filename
        sub_dirs.remove(".DS_Store")
        for each_sub in sub_dirs:
            # List files in the images folder
            image_files = os.listdir(os.path.join(images_folder, each_sub))
            for image_file in image_files:
                # Extract the name without the extension
                base_name = os.path.splitext(image_file)[0]

                # Construct the label file name by adding '_label' before the extension
                label_file = base_name + "_masks.tif"  # + os.path.splitext(image_file)[1]
                flows_file = base_name + "_flows.tif"
                # Check if the corresponding label file exists in the labels folder
                if label_file in os.listdir(os.path.join(images_folder, each_sub)):
                    # Add the file names to the training data list
                    training_data.append(
                        {
                            "image": os.path.join("omnipose_dataset", each_op, "train_sorted", each_sub, image_file),
                            "label": os.path.join("omnipose_dataset", each_op, "train_sorted", each_sub, label_file),
                            "flows": os.path.join("omnipose_dataset", each_op, "train_sorted", each_sub, flows_file),
                        }
                    )

        # Convert data_pairs to a numpy array for easy indexing by KFold
        data_pairs_array = np.array(training_data)

        # Initialize KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Assign fold numbers
        for fold, (train_index, val_index) in enumerate(kf.split(data_pairs_array)):
            for idx in val_index:
                data_pairs_array[idx]["fold"] = fold

        # Convert the array back to a list and sort by fold
        sorted_data_pairs = sorted(data_pairs_array.tolist(), key=lambda x: x["fold"])

        # Initialize the list for testing data
        testing_data = []

        test_sub_dirs = os.listdir(test_images_folder)
        # Likely Omnipose dataset was created using a Mac and hence the spare filename
        test_sub_dirs.remove(".DS_Store")
        # Loop through each image file to find its corresponding label file
        for each_test_sub in test_sub_dirs:
            # List files in the images folder
            test_image_files = os.listdir(os.path.join(test_images_folder, each_test_sub))
            for image_file in test_image_files:
                # Extract the name without the extension
                base_name = os.path.splitext(image_file)[0]

                # Construct the label file name by adding '_label' before the extension
                label_file = base_name + "_masks.tif"  # + os.path.splitext(image_file)[1]

                # Check if the corresponding label file exists in the labels folder
                if label_file in os.listdir(os.path.join(test_images_folder, each_test_sub)):
                    # Add the file names to the training data list
                    testing_data.append(
                        {
                            "image": os.path.join(
                                "omnipose_dataset", each_op, "test_sorted", each_test_sub, image_file
                            ),
                            "label": os.path.join(
                                "omnipose_dataset", each_op, "test_sorted", each_test_sub, label_file
                            ),
                        }
                    )

        all_data = {}
        all_data["training"] = sorted_data_pairs
        all_data["testing"] = testing_data

        # Save the training data list to a JSON file
        with open(json_f_path, "w") as json_file:
            json.dump(all_data, json_file, indent=4)


def nips_json_file(dataset_dir, json_f_path):
    """
    This function takes in the directory of extracted NIPS cell segmentation challenge as input
    and creates a json list with 5 folds. Separate testing set is recorded in the json list.
    Please note that there are some hard-coded directory names as per the original dataset.
    At the time of creation, the NIPS zipfile had 'Training-labeled' and 'Testing' directories that
    both contained 'images' and 'labels' directories
    """
    # The directory containing your files
    json_save_path = os.path.normpath(json_f_path)
    directory = os.path.join(dataset_dir, "Training-labeled")
    test_directory = os.path.join(dataset_dir, "Testing", "Public")
    # List to hold all image-mask pairs
    data_pairs = []
    test_data_pairs = []
    all_data = {}

    # Scan the directory for image files and create pairs
    for filename in os.listdir(os.path.join(directory, "images")):
        if os.path.exists(os.path.join(directory, "images", filename)):
            # Extract the name without the extension
            base_name = os.path.splitext(filename)[0]

            # Construct the label file name by adding '_label' before the extension
            label_file = base_name + "_label.tiff"  # + os.path.splitext(image_file)[1]

            # Check if the corresponding label file exists in the labels folder
            if label_file in os.listdir(os.path.join(directory, "labels")):
                # Add the file names to the training data list
                data_pairs.append(
                    {
                        "image": os.path.join("nips_dataset", "Training-labeled", "images", filename),
                        "label": os.path.join("nips_dataset", "Training-labeled", "labels", label_file),
                    }
                )

    # Convert data_pairs to a numpy array for easy indexing by KFold
    data_pairs_array = np.array(data_pairs)

    # Initialize KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Assign fold numbers
    for fold, (train_index, val_index) in enumerate(kf.split(data_pairs_array)):
        for idx in val_index:
            data_pairs_array[idx]["fold"] = fold

    # Convert the array back to a list and sort by fold
    sorted_data_pairs = sorted(data_pairs_array.tolist(), key=lambda x: x["fold"])

    print(sorted_data_pairs)

    # Scan the directory for image files and create pairs
    for filename in os.listdir(os.path.join(test_directory, "images")):
        if os.path.exists(os.path.join(test_directory, "images", filename)):
            # Extract the name without the extension
            base_name = os.path.splitext(filename)[0]

            # Construct the label file name by adding '_label' before the extension
            label_file = base_name + "_label.tiff"  # + os.path.splitext(image_file)[1]

            # Check if the corresponding label file exists in the labels folder
            if label_file in os.listdir(os.path.join(test_directory, "labels")):
                # Add the file names to the training data list
                test_data_pairs.append(
                    {
                        "image": os.path.join("nips_dataset", "Testing", "Public", "images", filename),
                        "label": os.path.join("nips_dataset", "Testing", "Public", "labels", label_file),
                    }
                )

    all_data["training"] = sorted_data_pairs
    all_data["testing"] = test_data_pairs

    with open(json_save_path, "w") as j_file:
        json.dump(all_data, j_file, indent=4)
    j_file.close()


def kaggle_json_file(dataset_dir, json_f_path):
    """
    This function takes in the directory of kaggle nuclei extracted dataset as input and
    creates a json list with 5 folds.
    Please note that there are some hard-coded directory names as per the original dataset.
    The function creates an instance processed dataset and then a 5 fold json file based on
    the instance processed dataset
    """
    data_dir = os.path.join(dataset_dir, "stage1_train")
    saving_path = os.path.join(dataset_dir, "instance_processed_data")
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)

    # Process the images and create instance masks first
    for idx, subdir in enumerate(os.listdir(data_dir)):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            images_dir = os.path.join(subdir_path, "images")
            masks_dir = os.path.join(subdir_path, "masks")
            if os.path.isdir(images_dir) and os.path.isdir(masks_dir):
                image_file = os.path.join(images_dir, os.listdir(images_dir)[0])
                filename_prefix = f"kg_bowl_{idx}_"

                mask_data = concatenate_masks(masks_dir)

                # ## Apply channel-wise normalization and use only the first three channels
                # image_data = imageio.imread(image_file)
                # normalized_image = normalize_image(image_data[..., :3])
                # imageio.imwrite(os.path.join(saving_path, f"{filename_prefix}img.tiff"), normalized_image)
                shutil.copyfile(image_file, os.path.join(saving_path, f"{filename_prefix}img.png"))
                imageio.imwrite(os.path.join(saving_path, f"{filename_prefix}img_masks.tiff"), mask_data)

    directory = saving_path

    # List to hold all image-mask pairs
    data_pairs = []
    all_data = {}

    # Scan the directory for image files and create pairs
    for filename in os.listdir(directory):
        if filename.endswith("_img.png"):
            # Construct the corresponding mask filename
            mask_filename = filename.replace("_img.png", "_img_masks.tiff")

            # Check if the corresponding mask file exists
            if os.path.exists(os.path.join(directory, mask_filename)):
                # Add the pair to the list
                data_pairs.append(
                    {
                        "image": os.path.join("kaggle_dataset", "instance_processed_data", filename),
                        "label": os.path.join("kaggle_dataset", "instance_processed_data", mask_filename),
                    }
                )

    # Convert data_pairs to a numpy array for easy indexing by KFold
    data_pairs_array = np.array(data_pairs)

    # Initialize KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Assign fold numbers
    for fold, (train_index, val_index) in enumerate(kf.split(data_pairs_array)):
        for idx in val_index:
            data_pairs_array[idx]["fold"] = fold

    # Convert the array back to a list and sort by fold
    sorted_data_pairs = sorted(data_pairs_array.tolist(), key=lambda x: x["fold"])

    print(sorted_data_pairs)

    all_data["training"] = sorted_data_pairs

    with open(json_f_path, "w") as j_file:
        json.dump(all_data, j_file, indent=4)
    j_file.close()


def deepbacs_json_file(dataset_dir, json_f_path):
    """
    This function takes in the directory of deepbacs extracted dataset as input and
    creates a json list with 5 folds. Separate testing set is recorded in the json list.
    Please note that there are some hard-coded directory names as per the original dataset.
    At the time of creation, the deepbacs zipfile had 'training' and 'test' directories that
    both contained 'source' and 'target' directories
    """
    # The directory containing your files
    json_save_path = os.path.normpath(json_f_path)
    directory = os.path.join(dataset_dir, "training")
    test_directory = os.path.join(dataset_dir, "test")
    # List to hold all image-mask pairs
    data_pairs = []
    test_data_pairs = []
    all_data = {}

    # Scan the directory for image files and create pairs
    for filename in os.listdir(os.path.join(directory, "source")):
        if os.path.exists(os.path.join(directory, "source", filename)):
            # Construct the corresponding mask filename
            mask_filename = filename

            # Check if the corresponding mask file exists
            if os.path.exists(os.path.join(directory, "target", mask_filename)):
                # Add the pair to the list
                data_pairs.append(
                    {
                        "image": os.path.join("deepbacs_dataset", "training", "source", filename),
                        "label": os.path.join("deepbacs_dataset", "training", "target", mask_filename),
                    }
                )

    # Convert data_pairs to a numpy array for easy indexing by KFold
    data_pairs_array = np.array(data_pairs)

    # Initialize KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Assign fold numbers
    for fold, (train_index, val_index) in enumerate(kf.split(data_pairs_array)):
        for idx in val_index:
            data_pairs_array[idx]["fold"] = fold

    # Convert the array back to a list and sort by fold
    sorted_data_pairs = sorted(data_pairs_array.tolist(), key=lambda x: x["fold"])

    print(sorted_data_pairs)

    # Scan the directory for image files and create pairs
    for filename in os.listdir(os.path.join(test_directory, "source")):
        if os.path.exists(os.path.join(test_directory, "source", filename)):
            # Construct the corresponding mask filename
            mask_filename = filename

            # Check if the corresponding mask file exists
            if os.path.exists(os.path.join(test_directory, "target", mask_filename)):
                # Add the pair to the list
                test_data_pairs.append(
                    {
                        "image": os.path.join("deepbacs_dataset", "test", "source", filename),
                        "label": os.path.join("deepbacs_dataset", "test", "target", filename),
                    }
                )

    all_data["training"] = sorted_data_pairs
    all_data["testing"] = test_data_pairs

    with open(json_save_path, "w") as j_file:
        json.dump(all_data, j_file, indent=4)
    j_file.close()


def cellpose_json_file(dataset_dir, json_f_path):
    """
    This function takes in the directory of cellpose extracted dataset as input and
    creates a json list with 5 folds. Separate testing set is recorded in the json list.
    Please note that there are some hard-coded directory names as per the original dataset.
    At the time of creation, the cellpose dataset had 'train.zip' and 'test.zip' that
    extracted as 'train' and 'test' directories
    """
    # The directory containing your files
    json_save_path = os.path.normpath(json_f_path)
    directory = os.path.join(dataset_dir, "train")
    test_directory = os.path.join(dataset_dir, "test")

    # List to hold all image-mask pairs
    data_pairs = []
    test_data_pairs = []
    all_data = {}

    # Scan the directory for image files and create pairs
    for filename in os.listdir(directory):
        if filename.endswith("_img.png"):
            # Construct the corresponding mask filename
            mask_filename = filename.replace("_img.png", "_masks.png")

            # Check if the corresponding mask file exists
            if os.path.exists(os.path.normpath(os.path.join(directory, mask_filename))):
                # Add the pair to the list
                data_pairs.append(
                    {
                        "image": os.path.join("cellpose_dataset", "train", filename),
                        "label": os.path.join("cellpose_dataset", "train", mask_filename),
                    }
                )

    # Convert data_pairs to a numpy array for easy indexing by KFold
    data_pairs_array = np.array(data_pairs)

    # Initialize KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Assign fold numbers
    for fold, (train_index, val_index) in enumerate(kf.split(data_pairs_array)):
        for idx in val_index:
            data_pairs_array[idx]["fold"] = fold

    # Convert the array back to a list and sort by fold
    sorted_data_pairs = sorted(data_pairs_array.tolist(), key=lambda x: x["fold"])

    print(sorted_data_pairs)

    # Scan the directory for image files and create pairs
    for filename in os.listdir(test_directory):
        if filename.endswith("_img.png"):
            # Construct the corresponding mask filename
            mask_filename = filename.replace("_img.png", "_masks.png")

            # Check if the corresponding mask file exists
            if os.path.exists(os.path.join(directory, mask_filename)):
                # Add the pair to the list
                test_data_pairs.append(
                    {
                        "image": os.path.join("cellpose_dataset", "test", filename),
                        "label": os.path.join("cellpose_dataset", "test", mask_filename),
                    }
                )

    all_data["training"] = sorted_data_pairs
    all_data["testing"] = test_data_pairs

    with open(json_save_path, "w") as j_file:
        json.dump(all_data, j_file, indent=4)
    j_file.close()


def extract_zip(zip_path, extract_to):
    # Ensure the target directory exists
    print(f"Extracting from: {zip_path}")
    print(f"Extracting to: {extract_to}")

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Extract all contents of the zip file to the specified directory
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--dir", type=str, help="Directory of datasets to generate json", default="/set/the/path")

    args = parser.parse_args()
    data_root_path = os.path.normpath(args.dir)

    if not os.path.exists(os.path.join(data_root_path, "json_files")):
        os.mkdir(os.path.join(data_root_path, "json_files"))

    dataset_dict = {
        "cellpose_dataset": ["train.zip", "test.zip"],
        "deepbacs_dataset": ["deepbacs.zip"],
        "kaggle_dataset": ["data-science-bowl-2018.zip"],
        "nips_dataset": ["nips_train.zip", "nips_test.zip"],
        "omnipose_dataset": ["datasets.zip"],
        "tissuenet_dataset": ["tissuenet_v1.0.zip"],
        "livecell_dataset": [
            "livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images_per_celltype.zip"
        ],
    }

    for key, value in dataset_dict.items():
        dataset_path = os.path.join(data_root_path, key)

        for each_zipped in value:
            in_path = os.path.join(dataset_path, each_zipped)
            try:
                if os.path.exists(in_path):
                    print(f"File exists at: {in_path}")
            except:
                print(f"File: {in_path} was not found")
            out_path = os.path.join(dataset_path)
            extract_zip(in_path, out_path)

    print(
        "If we reached here, that means all zip files got extracted ... Working on pre-processing and generating json files"
    )

    # Looping over all datasets again, Cellpose & Deepbacs have a similar directory structure
    for key, value in dataset_dict.items():
        if key == "cellpose_dataset":
            print("Creating Cellpose Dataset Json file ...")
            dataset_path = os.path.join(data_root_path, key)
            json_path = os.path.join(data_root_path, "json_files", "cellpose.json")
            cellpose_json_file(dataset_dir=dataset_path, json_f_path=json_path)

        elif key == "nips_dataset":
            print("Creating NIPS Dataset Json file ...")
            dataset_path = os.path.join(data_root_path, key)
            json_path = os.path.join(data_root_path, "json_files", "nips.json")
            nips_json_file(dataset_dir=dataset_path, json_f_path=json_path)

        elif key == "omnipose_dataset":
            print("Creating Omnipose Dataset Json files ...")
            dataset_path = os.path.join(data_root_path, key)
            json_path = os.path.join(data_root_path, "json_files")
            omnipose_json_file(dataset_dir=dataset_path, json_path=json_path)

        elif key == "kaggle_dataset":
            print("Needs additional extraction")
            train_zip_path = os.path.join(data_root_path, key, "stage1_train.zip")
            zip_out_path = os.path.join(data_root_path, key, "stage1_train")
            extract_zip(train_zip_path, zip_out_path)
            print("Creating Kaggle Dataset Json files ...")
            dataset_path = os.path.join(data_root_path, key)
            json_f_path = os.path.join(data_root_path, "json_files", "kaggle.json")
            kaggle_json_file(dataset_dir=dataset_path, json_f_path=json_f_path)

        elif key == "livecell_dataset":
            print("Creating LiveCell Dataset Json files ... Please note that 7 files will be created from livecell")
            dataset_path = os.path.join(data_root_path, key)
            json_base_name = os.path.join(data_root_path, "json_files")
            livecell_json_files(dataset_dir=dataset_path, json_f_path=json_base_name)

        elif key == "deepbacs_dataset":
            print("Creating Deepbacs Dataset Json file ...")
            dataset_path = os.path.join(data_root_path, key)
            json_path = os.path.join(data_root_path, "json_files", "deepbacs.json")
            deepbacs_json_file(dataset_dir=dataset_path, json_f_path=json_path)

        elif key == "tissuenet_dataset":
            print("Creating TissueNet Dataset Json files ... Please note that 13 files will be created from tissuenet")
            dataset_path = os.path.join(data_root_path, key)
            json_base_name = os.path.join(data_root_path, "json_files")
            tissuenet_json_files(dataset_dir=dataset_path, json_f_path=json_base_name)

    return None


if __name__ == "__main__":
    main()
