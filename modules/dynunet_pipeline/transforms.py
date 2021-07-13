import numpy as np
from monai.transforms import (CastToTyped,
                              Compose, CropForegroundd, EnsureChannelFirstd, LoadImaged,
                              NormalizeIntensity, RandCropByPosNegLabeld,
                              RandFlipd, RandGaussianNoised,
                              RandGaussianSmoothd, RandScaleIntensityd,
                              RandZoomd, SpatialCrop, SpatialPadd, EnsureTyped)
from monai.transforms.compose import MapTransform
from monai.transforms.utils import generate_spatial_bounding_box
from skimage.transform import resize

from task_params import clip_values, normalize_values, patch_size, spacing


def get_task_transforms(mode, task_id, pos_sample_num, neg_sample_num, num_samples):
    if mode != "test":
        keys = ["image", "label"]
    else:
        keys = ["image"]

    load_transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
    ]
    # 2. sampling
    sample_transforms = [
        PreprocessAnisotropic(
            keys=keys,
            clip_values=clip_values[task_id],
            pixdim=spacing[task_id],
            normalize_values=normalize_values[task_id],
            model_mode=mode,
        ),
    ]
    # 3. spatial transforms
    if mode == "train":
        other_transforms = [
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size[task_id]),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size[task_id],
                pos=pos_sample_num,
                neg=neg_sample_num,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandZoomd(
                keys=["image", "label"],
                min_zoom=0.9,
                max_zoom=1.2,
                mode=("trilinear", "nearest"),
                align_corners=(True, None),
                prob=0.15,
            ),
            RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.15),
                sigma_y=(0.5, 1.15),
                sigma_z=(0.5, 1.15),
                prob=0.15,
            ),
            RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15),
            RandFlipd(["image", "label"], spatial_axis=[0], prob=0.5),
            RandFlipd(["image", "label"], spatial_axis=[1], prob=0.5),
            RandFlipd(["image", "label"], spatial_axis=[2], prob=0.5),
            CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8)),
            EnsureTyped(keys=["image", "label"]),
        ]
    elif mode == "validation":
        other_transforms = [
            CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8)),
            EnsureTyped(keys=["image", "label"]),
        ]
    else:
        other_transforms = [
            CastToTyped(keys=["image"], dtype=(np.float32)),
            EnsureTyped(keys=["image"]),
        ]

    all_transforms = load_transforms + sample_transforms + other_transforms
    return Compose(all_transforms)


def resample_image(image, shape, anisotrophy_flag):
    resized_channels = []
    if anisotrophy_flag:
        for image_c in image:
            resized_slices = []
            for i in range(image_c.shape[-1]):
                image_c_2d_slice = image_c[:, :, i]
                image_c_2d_slice = resize(
                    image_c_2d_slice,
                    shape[:-1],
                    order=3,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                resized_slices.append(image_c_2d_slice)
            resized = np.stack(resized_slices, axis=-1)
            resized = resize(
                resized,
                shape,
                order=0,
                mode="constant",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            resized_channels.append(resized)
    else:
        for image_c in image:
            resized = resize(
                image_c,
                shape,
                order=3,
                mode="edge",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            resized_channels.append(resized)
    resized = np.stack(resized_channels, axis=0)
    return resized


def resample_label(label, shape, anisotrophy_flag):
    reshaped = np.zeros(shape, dtype=np.uint8)
    n_class = np.max(label)
    if anisotrophy_flag:
        shape_2d = shape[:-1]
        depth = label.shape[-1]
        reshaped_2d = np.zeros((*shape_2d, depth), dtype=np.uint8)

        for class_ in range(1, int(n_class) + 1):
            for depth_ in range(depth):
                mask = label[0, :, :, depth_] == class_
                resized_2d = resize(
                    mask.astype(float),
                    shape_2d,
                    order=1,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                reshaped_2d[:, :, depth_][resized_2d >= 0.5] = class_
        for class_ in range(1, int(n_class) + 1):
            mask = reshaped_2d == class_
            resized = resize(
                mask.astype(float),
                shape,
                order=0,
                mode="constant",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            reshaped[resized >= 0.5] = class_
    else:
        for class_ in range(1, int(n_class) + 1):
            mask = label[0] == class_
            resized = resize(
                mask.astype(float),
                shape,
                order=1,
                mode="edge",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            reshaped[resized >= 0.5] = class_

    reshaped = np.expand_dims(reshaped, 0)
    return reshaped


def recovery_prediction(prediction, shape, anisotrophy_flag):
    reshaped = np.zeros(shape, dtype=np.uint8)
    n_class = shape[0]
    if anisotrophy_flag:
        c, h, w = prediction.shape[:-1]
        d = shape[-1]
        reshaped_d = np.zeros((c, h, w, d), dtype=np.uint8)
        for class_ in range(1, n_class):
            mask = prediction[class_] == 1
            resized_d = resize(
                mask.astype(float),
                (h, w, d),
                order=0,
                mode="constant",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            reshaped_d[class_][resized_d >= 0.5] = 1

        for class_ in range(1, n_class):
            for depth_ in range(d):
                mask = reshaped_d[class_, :, :, depth_] == 1
                resized_hw = resize(
                    mask.astype(float),
                    shape[1:-1],
                    order=1,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                reshaped[class_, :, :, depth_][resized_hw >= 0.5] = 1
    else:
        for class_ in range(1, n_class):
            mask = prediction[class_] == 1
            resized = resize(
                mask.astype(float),
                shape[1:],
                order=1,
                mode="edge",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            reshaped[class_][resized >= 0.5] = 1

    return reshaped


class PreprocessAnisotropic(MapTransform):
    """
        This transform class takes NNUNet's preprocessing method for reference.
        That code is in:
        https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py

    """

    def __init__(
        self,
        keys,
        clip_values,
        pixdim,
        normalize_values,
        model_mode,
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.low = clip_values[0]
        self.high = clip_values[1]
        self.target_spacing = pixdim
        self.mean = normalize_values[0]
        self.std = normalize_values[1]
        self.training = False
        self.crop_foreg = CropForegroundd(keys=["image", "label"], source_key="image")
        self.normalize_intensity = NormalizeIntensity(nonzero=True, channel_wise=True)
        if model_mode in ["train"]:
            self.training = True

    def calculate_new_shape(self, spacing, shape):
        spacing_ratio = np.array(spacing) / np.array(self.target_spacing)
        new_shape = (spacing_ratio * np.array(shape)).astype(int).tolist()
        return new_shape

    def check_anisotrophy(self, spacing):
        def check(spacing):
            return np.max(spacing) / np.min(spacing) >= 3

        return check(spacing) or check(self.target_spacing)

    def __call__(self, data):
        # load data
        d = dict(data)
        image = d["image"]
        image_spacings = d["image_meta_dict"]["pixdim"][1:4].tolist()

        if "label" in self.keys:
            label = d["label"]
            label[label < 0] = 0

        if self.training:
            # only task 04 does not be impacted
            cropped_data = self.crop_foreg({"image": image, "label": label})
            image, label = cropped_data["image"], cropped_data["label"]
        else:
            d["original_shape"] = np.array(image.shape[1:])
            box_start, box_end = generate_spatial_bounding_box(image)
            image = SpatialCrop(roi_start=box_start, roi_end=box_end)(image)
            d["bbox"] = np.vstack([box_start, box_end])
            d["crop_shape"] = np.array(image.shape[1:])

        original_shape = image.shape[1:]
        # calculate shape
        resample_flag = False
        anisotrophy_flag = False
        if self.target_spacing != image_spacings:
            # resample
            resample_flag = True
            resample_shape = self.calculate_new_shape(image_spacings, original_shape)
            anisotrophy_flag = self.check_anisotrophy(image_spacings)
            image = resample_image(image, resample_shape, anisotrophy_flag)
            if self.training:
                label = resample_label(label, resample_shape, anisotrophy_flag)

        d["resample_flag"] = resample_flag
        d["anisotrophy_flag"] = anisotrophy_flag
        # clip image for CT dataset
        if self.low != 0 or self.high != 0:
            image = np.clip(image, self.low, self.high)
            image = (image - self.mean) / self.std
        else:
            image = self.normalize_intensity(image.copy())

        d["image"] = image

        if "label" in self.keys:
            d["label"] = label

        return d
