import argparse
import gc
import os
import shutil
import time
import warnings
import zipfile

import imageio.v3 as imageio
import numpy as np
from PIL import Image
from pycocotools.coco import COCO


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


def livecell_process_files(dataset_dir):
    """
    This function takes in the directory of livecell extracted dataset as input and
    extracts labels from the coco format.
    """

    # "A172", "BT474", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"
    # "BV2" is being skipped, runs into memory constraints
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


def tissuenet_process_files(dataset_dir):
    """
    This function takes in the directory of TissueNet extracted dataset as input and
    creates tiled images into 4 from each image
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


def kaggle_process_files(dataset_dir):
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
    parser = argparse.ArgumentParser(description="Script to process the cell imaging datasets")
    parser.add_argument("--dir", type=str, help="Directory of datasets to process it ...", default="/set/the/path")

    args = parser.parse_args()
    data_root_path = os.path.normpath(args.dir)

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

    print("If we reached here, that means all zip files got extracted ... Working on pre-processing")

    # Looping over all datasets again, Cellpose & Deepbacs have a similar directory structure
    for key, value in dataset_dict.items():
        if key == "kaggle_dataset":
            print("Needs additional extraction")
            train_zip_path = os.path.join(data_root_path, key, "stage1_train.zip")
            zip_out_path = os.path.join(data_root_path, key, "stage1_train")
            extract_zip(train_zip_path, zip_out_path)
            print("Processing Kaggle Dataset ...")
            dataset_path = os.path.join(data_root_path, key)
            kaggle_process_files(dataset_dir=dataset_path)

        elif key == "livecell_dataset":
            print("Processing LiveCell Dataset ...")
            print(
                "Fyi, this processing might take upto an hour, coffee break might be more fruitful in the meanwhile ..."
            )
            dataset_path = os.path.join(data_root_path, key)
            livecell_process_files(dataset_dir=dataset_path)

        elif key == "tissuenet_dataset":
            print("Processing TissueNet Dataset ...")
            dataset_path = os.path.join(data_root_path, key)
            tissuenet_process_files(dataset_dir=dataset_path)

    return None


if __name__ == "__main__":
    main()
