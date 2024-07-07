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

import random
from datetime import datetime
import json
import time


import monai
import torch
from monai.transforms import Compose, SaveImage
from monai.data import MetaTensor
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from tqdm import tqdm
from generative.metrics import FIDMetric
from generative.inferers import LatentDiffusionInferer
from .utils import binarize_labels, MapLabelValue, general_mask_generation_post_process, get_body_region_index_from_mask
from .find_masks import find_masks
from .augmentation import augmentation


class ReconModel(torch.nn.Module):
    def __init__(self, autoencoder, scale_factor):
        super().__init__()
        self.autoencoder = autoencoder
        self.scale_factor = scale_factor

    def forward(self, z):
        recon_pt_nda = self.autoencoder.decode_stage_2_outputs(z / self.scale_factor)
        return recon_pt_nda


def ldm_conditional_sample_one_mask(
    autoencoder,
    diffusion_unet,
    noise_scheduler,
    scale_factor,
    anatomy_size,
    device,
    latent_shape,
    label_dict_remap_json,
    num_inference_steps=1000,
):
    with torch.no_grad(), torch.cuda.amp.autocast():

        # Generate random noise
        latents = (
            torch.randn(
                [
                    1,
                ]
                + list(latent_shape)
            )
            .half()
            .to(device)
        )
        anatomy_size = torch.FloatTensor(anatomy_size).unsqueeze(0).unsqueeze(0).half().to(device)
        # synthesize masks
        noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        inferer_ddpm = LatentDiffusionInferer(noise_scheduler, scale_factor=scale_factor)
        synthetic_mask = inferer_ddpm.sample(
            input_noise=latents,
            autoencoder_model=autoencoder,
            diffusion_model=diffusion_unet,
            scheduler=noise_scheduler,
            verbose=True,
            conditioning=anatomy_size.to(device),
        )
        synthetic_mask = torch.softmax(synthetic_mask, dim=1)
        synthetic_mask = torch.argmax(synthetic_mask, dim=1, keepdim=True)
        # mapping raw index to 132 labels
        with open(label_dict_remap_json, "r") as f:
            mapping_dict = json.load(f)
        mapping = [v for _, v in mapping_dict.items()]
        mapper = MapLabelValue(
            orig_labels=[pair[0] for pair in mapping],
            target_labels=[pair[1] for pair in mapping],
            dtype=torch.uint8,
        )
        synthetic_mask = mapper(synthetic_mask[0, ...])[None, ...].to(device)

        ###### post process #####
        data = synthetic_mask.squeeze().cpu().detach().numpy()

        labels = [23, 24, 26, 27, 128]
        target_tumor_label = None
        for index, size in enumerate(anatomy_size[5:10]):
            if size.item() != -1.0:
                target_tumor_label = labels[index]

        print("target_tumor_label for postprocess:", target_tumor_label)
        data = general_mask_generation_post_process(data, target_tumor_label=target_tumor_label, device=device)
        synthetic_mask = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(device)

    return synthetic_mask


def ldm_conditional_sample_one_image(
    autoencoder,
    diffusion_unet,
    controlnet,
    noise_scheduler,
    scale_factor,
    device,
    comebine_label_or,
    top_region_index_tensor,
    bottom_region_index_tensor,
    spacing_tensor,
    latent_shape,
    output_size,
    noise_factor,
    num_inference_steps=1000,
):
    # CT image intensity range
    a_min = -1000
    a_max = 1000
    # autoencoder output intensity range
    b_min = 0.0
    b_max = 1

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        print("Start generating latent features...")
        start_time = time.time()
        # generate segmentation mask
        comebine_label = comebine_label_or.to(device)
        if (
            output_size[0] != comebine_label.shape[2]
            or output_size[1] != comebine_label.shape[3]
            or output_size[2] != comebine_label.shape[4]
        ):
            print(
                "output_size is not a desired value. Need to interpolate the mask to match with output_size. The result image will be very low quality."
            )
            comebine_label = torch.nn.functional.interpolate(comebine_label, size=output_size, mode="nearest")

        controlnet_cond_vis = binarize_labels(comebine_label.as_tensor().long()).half()

        # Generate random noise
        latents = (
            torch.randn(
                [
                    1,
                ]
                + list(latent_shape)
            )
            .half()
            .to(device)
            * noise_factor
        )

        # synthesize latents
        noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        for t in tqdm(noise_scheduler.timesteps, ncols=110):
            # Get controlnet output
            down_block_res_samples, mid_block_res_sample = controlnet(
                x=latents,
                timesteps=torch.Tensor((t,)).to(device),
                controlnet_cond=controlnet_cond_vis,
            )
            latent_model_input = latents
            noise_pred = diffusion_unet(
                x=latent_model_input,
                timesteps=torch.Tensor((t,)).to(device),
                top_region_index_tensor=top_region_index_tensor,
                bottom_region_index_tensor=bottom_region_index_tensor,
                spacing_tensor=spacing_tensor,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )
            latents, _ = noise_scheduler.step(noise_pred, t, latents)
        end_time = time.time()
        print(f"Latent features generation time: {end_time - start_time} seconds")

        # decode latents to synthesized images
        print("Start decoding latent features into images...")
        start_time = time.time()
        synthetic_images = sliding_window_inference(
            inputs=latents,
            roi_size=(
                min(output_size[0] // 4 // 4 * 3, 96),
                min(output_size[1] // 4 // 4 * 3, 96),
                min(output_size[2] // 4 // 4 * 3, 96),
            ),
            sw_batch_size=1,
            predictor=recon_model,
            mode="gaussian",
            overlap=2.0 / 3.0,
            sw_device=device,
            device=device,
        )
        synthetic_images = torch.clip(synthetic_images, b_min, b_max).cpu()
        end_time = time.time()
        print(f"Image decoding time: {end_time - start_time} seconds")

        ## post processing:
        # project output to [0, 1]
        synthetic_images = (synthetic_images - b_min) / (b_max - b_min)
        # project output to [-1000, 1000]
        synthetic_images = synthetic_images * (a_max - a_min) + a_min
        # regularize background intensities
        synthetic_images = crop_img_body_mask(synthetic_images, comebine_label)

    return synthetic_images, comebine_label


def filter_mask_with_organs(comebine_label, anatomy_list):
    # final output mask file has shape of output_size, contaisn labels in anatomy_list
    # it is already interpolated to target size
    comebine_label = comebine_label.long()
    # filter out the organs that are not in anatomy_list
    for i in range(len(anatomy_list)):
        organ = anatomy_list[i]
        # replace it with a negative value so it will get mixed
        comebine_label[comebine_label == organ] = -(i + 1)
    # zero-out voxels with value not in anatomy_list
    comebine_label[comebine_label > 0] = 0
    # output positive values
    comebine_label = -comebine_label
    return comebine_label


def crop_img_body_mask(synthetic_images, comebine_label):
    synthetic_images[comebine_label == 0] = -1000
    return synthetic_images


def check_input(
    body_region,
    anatomy_list,
    label_dict_json,
    output_size,
    spacing,
    controllable_anatomy_size=[("pancreas", 0.5)],
):
    # check output_size and spacing format
    if output_size[0] != output_size[1]:
        raise ValueError(f"The first two components of output_size need to be equal, yet got {output_size}.")
    if (output_size[0] not in [256, 384, 512]) or (output_size[2] not in [128, 256, 384, 512, 640, 768]):
        raise ValueError(
            f"The output_size[0] have to be chosen from [256, 384, 512], and output_size[2] have to be chosen from [128, 256, 384, 512, 640, 768], yet got {output_size}."
        )

    if spacing[0] != spacing[1]:
        raise ValueError(f"The first two components of spacing need to be equal, yet got {spacing}.")
    if spacing[0] < 0.5 or spacing[0] > 3.0 or spacing[2] < 0.5 or spacing[2] > 5.0:
        raise ValueError(
            f"spacing[0] have to be between 0.5 and 3.0 mm, spacing[2] have to be between 0.5 and 5.0 mm, yet got {spacing}."
        )

    # check controllable_anatomy_size format
    if len(controllable_anatomy_size) > 10:
        raise ValueError(
            f"The length of list controllable_anatomy_size has to be less than 10. Yet got length equal to {len(controllable_anatomy_size)}."
        )
    available_controllable_organ = [
        "liver",
        "gallbladder",
        "stomach",
        "pancreas",
        "colon",
    ]
    available_controllable_tumor = [
        "hepatic tumor",
        "bone lesion",
        "lung tumor",
        "colon cancer primaries",
        "pancreatic tumor",
    ]
    available_controllable_anatomy = available_controllable_organ + available_controllable_tumor
    controllable_tumor = []
    controllable_organ = []
    for controllable_anatomy_size_pair in controllable_anatomy_size:
        if controllable_anatomy_size_pair[0] not in available_controllable_anatomy:
            raise ValueError(
                f"The controllable_anatomy have to be chosen from {available_controllable_anatomy}, yet got {controllable_anatomy_size_pair[0]}."
            )
        if controllable_anatomy_size_pair[0] in available_controllable_tumor:
            controllable_tumor += [controllable_anatomy_size_pair[0]]
        if controllable_anatomy_size_pair[0] in available_controllable_organ:
            controllable_organ += [controllable_anatomy_size_pair[0]]
        if controllable_anatomy_size_pair[1] == -1:
            continue
        if controllable_anatomy_size_pair[1] < 0 or controllable_anatomy_size_pair[1] > 1.0:
            raise ValueError(
                f"The controllable size scale have to be between 0 and 1,0, or equal to -1, yet got {controllable_anatomy_size_pair[1]}."
            )
    if len(controllable_tumor + controllable_organ) != len(list(set(controllable_tumor + controllable_organ))):
        raise ValueError(f"Please do not repeat controllable_anatomy. Got {controllable_tumor + controllable_organ}.")
    if len(controllable_tumor) > 1:
        raise ValueError(f"Only one controllable tumor is supported. Yet got {controllable_tumor}.")

    if len(controllable_anatomy_size) > 0:
        print(
            f"controllable_anatomy_size is not empty. We will ignore body_region and anatomy_list and synthesize based on controllable_anatomy_size ({controllable_anatomy_size})."
        )
    else:
        print(
            f"controllable_anatomy_size is empty. We will synthesize based on body_region ({body_region}) and anatomy_list ({anatomy_list})."
        )
        # check body_region format
        available_body_region = [
            "head",
            "chest",
            "thorax",
            "abdomen",
            "pelvis",
            "lower",
        ]
        for region in body_region:
            if region not in available_body_region:
                raise ValueError(
                    f"The components in body_region have to be chosen from {available_body_region}, yet got {region}."
                )

        # check anatomy_list format
        with open(label_dict_json) as f:
            label_dict = json.load(f)
        for anatomy in anatomy_list:
            if anatomy not in label_dict.keys():
                raise ValueError(
                    f"The components in anatomy_list have to be chosen from {label_dict.keys()}, yet got {anatomy}."
                )
    print(f"The generate results will have voxel size to be {spacing}mm, volume size to be {output_size}.")

    return


class LDMSampler:
    def __init__(
        self,
        body_region,
        anatomy_list,
        all_mask_files_json,
        all_anatomy_size_condtions_json,
        all_mask_files_base_dir,
        label_dict_json,
        label_dict_remap_json,
        autoencoder,
        diffusion_unet,
        controlnet,
        noise_scheduler,
        scale_factor,
        mask_generation_autoencoder,
        mask_generation_diffusion_unet,
        mask_generation_scale_factor,
        mask_generation_noise_scheduler,
        device,
        latent_shape,
        mask_generation_latent_shape,
        output_size,
        output_dir,
        controllable_anatomy_size,
        image_output_ext=".nii.gz",
        label_output_ext=".nii.gz",
        quality_check_args=None,
        spacing=[1, 1, 1],
        num_inference_steps=None,
        mask_generation_num_inference_steps=None,
        random_seed=None,
    ) -> None:

        if random_seed is not None:
            set_determinism(seed=random_seed)

        with open(label_dict_json, "r") as f:
            label_dict = json.load(f)
        self.all_anatomy_size_condtions_json = all_anatomy_size_condtions_json

        # intialize variables
        self.body_region = body_region
        self.anatomy_list = [label_dict[organ] for organ in anatomy_list]
        self.all_mask_files_json = all_mask_files_json
        self.data_root = all_mask_files_base_dir
        self.label_dict_remap_json = label_dict_remap_json
        self.autoencoder = autoencoder
        self.diffusion_unet = diffusion_unet
        self.controlnet = controlnet
        self.noise_scheduler = noise_scheduler
        self.scale_factor = scale_factor
        self.mask_generation_autoencoder = mask_generation_autoencoder
        self.mask_generation_diffusion_unet = mask_generation_diffusion_unet
        self.mask_generation_scale_factor = mask_generation_scale_factor
        self.mask_generation_noise_scheduler = mask_generation_noise_scheduler
        self.device = device
        self.latent_shape = latent_shape
        self.mask_generation_latent_shape = mask_generation_latent_shape
        self.output_size = output_size
        self.output_dir = output_dir
        self.noise_factor = 1.0
        self.controllable_anatomy_size = controllable_anatomy_size
        if len(self.controllable_anatomy_size):
            print("controllable_anatomy_size is given, mask generation is triggered!")
            # overwrite the anatomy_list by given organs in self.controllable_anatomy_size
            self.anatomy_list = [label_dict[organ_and_size[0]] for organ_and_size in self.controllable_anatomy_size]
        self.image_output_ext = image_output_ext
        self.label_output_ext = label_output_ext
        # Set the default value for number of inference steps to 1000
        self.num_inference_steps = num_inference_steps if num_inference_steps is not None else 1000
        self.mask_generation_num_inference_steps = (
            mask_generation_num_inference_steps if mask_generation_num_inference_steps is not None else 1000
        )

        # quality check disabled for this version
        self.quality_check_args = quality_check_args

        self.autoencoder.eval()
        self.diffusion_unet.eval()
        self.controlnet.eval()
        self.mask_generation_autoencoder.eval()
        self.mask_generation_diffusion_unet.eval()

        self.spacing = spacing

        self.val_transforms = Compose(
            [
                monai.transforms.LoadImaged(keys=["pseudo_label"]),
                monai.transforms.EnsureChannelFirstd(keys=["pseudo_label"]),
                monai.transforms.Orientationd(keys=["pseudo_label"], axcodes="RAS"),
                monai.transforms.EnsureTyped(keys=["pseudo_label"], dtype=torch.uint8),
                monai.transforms.Lambdad(keys="top_region_index", func=lambda x: torch.FloatTensor(x)),
                monai.transforms.Lambdad(keys="bottom_region_index", func=lambda x: torch.FloatTensor(x)),
                monai.transforms.Lambdad(keys="spacing", func=lambda x: torch.FloatTensor(x)),
                monai.transforms.Lambdad(keys="top_region_index", func=lambda x: x * 1e2),
                monai.transforms.Lambdad(keys="bottom_region_index", func=lambda x: x * 1e2),
                monai.transforms.Lambdad(keys="spacing", func=lambda x: x * 1e2),
            ]
        )
        print("LDM sampler initialized.")

    def sample_multiple_images(self, num_img):
        if len(self.controllable_anatomy_size) > 0:
            # we will use mask generation instead of finding candidate masks
            # create a dummy selected_mask_files for placeholder
            selected_mask_files = list(range(num_img))
            # prerpare organ size conditions
            anatomy_size_condtion = self.prepare_anatomy_size_condtion(self.controllable_anatomy_size)
        else:
            need_resample = False
            # find candidate mask and save to candidate_mask_files
            candidate_mask_files = find_masks(
                self.body_region,
                self.anatomy_list,
                self.spacing,
                self.output_size,
                True,
                self.all_mask_files_json,
                self.data_root,
            )
            if len(candidate_mask_files) < num_img:
                # if we cannot find enough masks based on the exact match of anatomy list, spacing, and output size,
                # then we will try to find the closest mask in terms of  spacing, and output size.
                print("Resample mask file to get desired output size and spacing")
                candidate_mask_files = self.find_closest_masks(num_img)
                need_resample = True

            selected_mask_files = self.select_mask(candidate_mask_files, num_img)
            print(f"Images will be generated based on {selected_mask_files}.")
            if len(selected_mask_files) != num_img:
                raise ValueError(
                    f"len(selected_mask_files) ({len(selected_mask_files)}) != num_img ({num_img}). This should not happen. Please revisit function select_mask(self, candidate_mask_files, num_img)."
                )
        for item in selected_mask_files:
            print("Prepare mask...")
            start_time = time.time()
            if len(self.controllable_anatomy_size) > 0:
                # generate a synthetic mask
                (
                    comebine_label_or,
                    top_region_index_tensor,
                    bottom_region_index_tensor,
                    spacing_tensor,
                ) = self.prepare_one_mask_and_meta_info(anatomy_size_condtion)
            else:
                # read in mask file
                mask_file = item["mask_file"]
                if_aug = item["if_aug"]
                (
                    comebine_label_or,
                    top_region_index_tensor,
                    bottom_region_index_tensor,
                    spacing_tensor,
                ) = self.read_mask_information(mask_file)
                if need_resample:
                    comebine_label_or = self.ensure_output_size_and_spacing(comebine_label_or)
                # mask augmentation
                if if_aug == True:
                    comebine_label_or = augmentation(comebine_label_or, self.output_size)
            end_time = time.time()
            print(f"Mask preparation time: {end_time - start_time} seconds.")
            torch.cuda.empty_cache()
            # generate image/label pairs
            to_generate = True
            try_time = 0
            while to_generate:
                synthetic_images, synthetic_labels = self.sample_one_pair(
                    comebine_label_or,
                    top_region_index_tensor,
                    bottom_region_index_tensor,
                    spacing_tensor,
                )
                # current quality always return True
                pass_quality_check = self.quality_check(synthetic_images)
                if pass_quality_check or try_time > 3:
                    # save image/label pairs
                    output_postfix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    synthetic_labels.meta["filename_or_obj"] = "sample.nii.gz"
                    synthetic_images = MetaTensor(synthetic_images, meta=synthetic_labels.meta)
                    img_saver = SaveImage(
                        output_dir=self.output_dir,
                        output_postfix=output_postfix + "_image",
                        output_ext=self.image_output_ext,
                        separate_folder=False,
                    )
                    img_saver(synthetic_images[0])
                    # filter out the organs that are not in anatomy_list
                    synthetic_labels = filter_mask_with_organs(synthetic_labels, self.anatomy_list)
                    label_saver = SaveImage(
                        output_dir=self.output_dir,
                        output_postfix=output_postfix + "_label",
                        output_ext=self.label_output_ext,
                        separate_folder=False,
                    )
                    label_saver(synthetic_labels[0])
                    to_generate = False
                else:
                    print("Generated image/label pair did not pass quality check, will re-generate another pair.")
                    try_time += 1
        return

    def select_mask(self, candidate_mask_files, num_img):
        selected_mask_files = []
        random.shuffle(candidate_mask_files)

        for n in range(num_img):
            mask_file = candidate_mask_files[n % len(candidate_mask_files)]
            selected_mask_files.append({"mask_file": mask_file, "if_aug": True})
        return selected_mask_files

    def sample_one_pair(
        self,
        comebine_label_or_aug,
        top_region_index_tensor,
        bottom_region_index_tensor,
        spacing_tensor,
    ):
        # generate image/label pairs
        synthetic_images, synthetic_labels = ldm_conditional_sample_one_image(
            autoencoder=self.autoencoder,
            diffusion_unet=self.diffusion_unet,
            controlnet=self.controlnet,
            noise_scheduler=self.noise_scheduler,
            scale_factor=self.scale_factor,
            device=self.device,
            comebine_label_or=comebine_label_or_aug,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
            latent_shape=self.latent_shape,
            output_size=self.output_size,
            noise_factor=self.noise_factor,
            num_inference_steps=self.num_inference_steps,
        )
        return synthetic_images, synthetic_labels

    def prepare_anatomy_size_condtion(
        self,
        controllable_anatomy_size,
    ):
        anatomy_size_idx = {
            "gallbladder": 0,
            "liver": 1,
            "stomach": 2,
            "pancreas": 3,
            "colon": 4,
            "lung tumor": 5,
            "pancreatic tumor": 6,
            "hepatic tumor": 7,
            "colon cancer primaries": 8,
            "bone lesion": 9,
        }
        provide_anatomy_size = [None for _ in range(10)]
        print("controllable_anatomy_size:", controllable_anatomy_size)
        for element in controllable_anatomy_size:
            anatomy_name, anatomy_size = element
            provide_anatomy_size[anatomy_size_idx[anatomy_name]] = anatomy_size

        with open(self.all_anatomy_size_condtions_json, "r") as f:
            all_anatomy_size_condtions = json.load(f)

        # loop through the database and find closest combinations
        candidate_list = []
        for anatomy_size in all_anatomy_size_condtions:
            size = anatomy_size["organ_size"]
            diff = 0
            for db_size, provide_size in zip(size, provide_anatomy_size):
                if provide_size is None:
                    continue
                diff += abs(provide_size - db_size)
            candidate_list.append((size, diff))
        candidate_condition = sorted(candidate_list, key=lambda x: x[1])[0][0]
        # print("provide_anatomy_size:", provide_anatomy_size)
        # print("candidate_condition:", candidate_condition)

        # overwrite the anatomy size provided by users
        for element in controllable_anatomy_size:
            anatomy_name, anatomy_size = element
            candidate_condition[anatomy_size_idx[anatomy_name]] = anatomy_size
        # print("final candidate_condition:", candidate_condition)
        return candidate_condition

    def prepare_one_mask_and_meta_info(self, anatomy_size_condtion):
        comebine_label_or = self.sample_one_mask(anatomy_size=anatomy_size_condtion)
        # TODO: current mask generation model only can generate 256^3 volumes with 1.5 mm spacing.
        affine = torch.zeros((4, 4))
        affine[0, 0] = 1.5
        affine[1, 1] = 1.5
        affine[2, 2] = 1.5
        affine[3, 3] = 1.0  # dummy
        comebine_label_or = MetaTensor(comebine_label_or, affine=affine)
        comebine_label_or = self.ensure_output_size_and_spacing(comebine_label_or)

        top_region_index, bottom_region_index = get_body_region_index_from_mask(comebine_label_or)

        spacing_tensor = torch.FloatTensor(self.spacing).unsqueeze(0).half().to(self.device) * 1e2
        top_region_index_tensor = torch.FloatTensor(top_region_index).unsqueeze(0).half().to(self.device) * 1e2
        bottom_region_index_tensor = torch.FloatTensor(bottom_region_index).unsqueeze(0).half().to(self.device) * 1e2

        return comebine_label_or, top_region_index_tensor, bottom_region_index_tensor, spacing_tensor

    def sample_one_mask(self, anatomy_size):
        # generate one synthetic mask
        synthetic_mask = ldm_conditional_sample_one_mask(
            self.mask_generation_autoencoder,
            self.mask_generation_diffusion_unet,
            self.mask_generation_noise_scheduler,
            self.mask_generation_scale_factor,
            anatomy_size,
            self.device,
            self.mask_generation_latent_shape,
            label_dict_remap_json=self.label_dict_remap_json,
            num_inference_steps=self.mask_generation_num_inference_steps,
        )
        return synthetic_mask

    def ensure_output_size_and_spacing(self, labels, check_contains_target_labels=True):
        current_spacing = [labels.affine[0, 0], labels.affine[1, 1], labels.affine[2, 2]]
        current_shape = list(labels.squeeze().shape)

        need_resample = False
        # check spacing
        for i, j in zip(current_spacing, self.spacing):
            if i != j:
                need_resample = True
        # check output size
        for i, j in zip(current_shape, self.output_size):
            if i != j:
                need_resample = True
        # resample to target size and spacing
        if need_resample:
            print("Resampling mask to target shape and spacing")
            print(f"Resize Spacing: {current_spacing} -> {self.spacing}")
            print(f"Output size: {current_shape} -> {self.output_size}")
            spacing = monai.transforms.Spacing(pixdim=tuple(self.spacing), mode="nearest")
            pad_crop = monai.transforms.ResizeWithPadOrCrop(spatial_size=tuple(self.output_size))
            labels = pad_crop(spacing(labels.squeeze(0))).unsqueeze(0).to(labels.dtype)

            contained_labels = torch.unique(labels)
            if check_contains_target_labels:
                # check if the resampled mask still contains those target labels
                for anatomy_label in self.anatomy_list:
                    if anatomy_label not in contained_labels:
                        raise ValueError(
                            f"Resampled mask does not contain required class labels {anatomy_label}. Please tune spacing and output size."
                        )
        return labels

    def read_mask_information(self, mask_file):
        val_data = self.val_transforms(mask_file)

        for key in [
            "pseudo_label",
            "spacing",
            "top_region_index",
            "bottom_region_index",
        ]:
            val_data[key] = val_data[key].unsqueeze(0).to(self.device)

        return (
            val_data["pseudo_label"],
            val_data["top_region_index"],
            val_data["bottom_region_index"],
            val_data["spacing"],
        )

    def find_closest_masks(self, num_img):
        # first check the database based on anatomy list
        candidates = find_masks(
            self.body_region,
            self.anatomy_list,
            self.spacing,
            self.output_size,
            False,
            self.all_mask_files_json,
            self.data_root,
        )

        if len(candidates) < num_img:
            raise ValueError(f"candidate masks are less than {num_img}).")
        # loop through the database and find closest combinations
        new_candidates = []
        for c in candidates:
            diff = 0
            for axis in range(3):
                # check diff in dim
                diff += abs((c["dim"][axis] - self.output_size[axis]) / 100)
                # check diff in spacing
                diff += abs(c["spacing"][axis] - self.spacing[axis])
            new_candidates.append((c, diff))
        # choose top-2*num_img candidates (at least 5)
        new_candidates = sorted(new_candidates, key=lambda x: x[1])[: max(2 * num_img, 5)]
        final_candidates = []
        # check top-2*num_img candidates and update spacing after resampling
        image_loader = monai.transforms.LoadImage(image_only=True, ensure_channel_first=True)
        for c, _ in new_candidates:
            label = image_loader(c["pseudo_label"])
            try:
                label = self.ensure_output_size_and_spacing(label.unsqueeze(0))
            except ValueError as e:
                if "Resampled mask does not contain required class labels" in str(e):
                    continue
                else:
                    raise e
            # get region_index after resample
            top_region_index, bottom_region_index = get_body_region_index_from_mask(label)
            c["top_region_index"] = top_region_index
            c["bottom_region_index"] = bottom_region_index
            c["spacing"] = self.spacing
            c["dim"] = self.output_size

            final_candidates.append(c)
        if len(final_candidates) == 0:
            raise ValueError("Cannot find body region with given organ list.")
        return final_candidates

    def quality_check(self, image):
        # This version disabled quality check
        return True
