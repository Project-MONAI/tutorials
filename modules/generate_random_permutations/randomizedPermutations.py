import numpy as np
import os
import torch
import glob
import random
from monai.data import DataLoader
from monai.transforms.transform import Transform
from monai.transforms import (Affine, LoadImage, Rotate, NormalizeIntensity, Transpose, Compose, Resize, AsChannelFirst, AsChannelLast, ScaleIntensity, RandFlip, Rotate90, AddChannel, GaussianSmooth, AdjustContrast)
from random import shuffle

class Dataset(torch.utils.data.Dataset):
	def __init__(self, image_file_list, transforms, shuffle_transforms=1):
		self.image_file_list = image_file_list
		if shuffle_transforms:
			transform_list = [LoadImage(image_only=True), AddChannel(), Resize((299, 299))] + shuffle(transforms)
			self.transform = Compose(transpose_list)
		else:
			self.transform = Compose([LoadImage(image_only=True), AddChannel(), Resize((299, 299))] + transforms)

	def __len__(self):
		return len(self.image_file_list)

	def __getitem__(self, index):
		return self.transform(self.image_file_list[index])


class AugmentData(object):
	def __init__(self, image_loading_transforms = [LoadImage(image_only=True)], augmentation_dict = {}, num_augmentations=5, output_size=(200, 200), batch_size=3):
		self.output_size = output_size
		self.batch_size = batch_size
		self.augmentation_dict = augmentation_dict
		self.aug_seq = self.create_augmentation_sequence()
		self.image_loading_transforms = image_loading_transforms
		self.num_augmentations = num_augmentations

	def create_augmentation_sequence(self):
		augmentation_transforms = []
		for aug, num_aug in self.augmentation_dict.items():
			_x = [aug]*num_aug
			augmentation_transforms = augmentation_transforms + _x
		return augmentation_transforms


	def create_transform_list(self, augmentation_sequence):
		transform_list = self.image_loading_transforms
		for _aug in augmentation_sequence:
			if _aug == 'rotate':
				transform_list.append(Rotate(random.randint(0, 100)))
			if _aug == 'flip':
				transform_list.append(RandFlip())
			if _aug == 'rotate90':
				transform_list.append(Rotate90())
			if _aug == 'intensityGaussian':
				transform_list.append(GaussianSmooth(sigma=random.randint(0, 10)))
			if _aug == 'adjustContrast':
				transform_list.append(AdjustContrast(gamma=random.randint(0, 10)))

		transform_list.append(ScaleIntensity())
		transform_list.append(Resize(self.output_size))
		return transform_list


	def create_native_transform_list(self):
		transform_list = Compose(self.image_loading_transforms + [ScaleIntensity(), Resize(self.output_size)])
		return transform_list


	def __call__(self, image_file_list, *args, **kwargs):
		image_file_list = image_file_list

		IMG = []
		for img in zip(image_file_list):
			native_transform_list = self.create_native_transform_list()
			native_img = native_transform_list(img)
			IMG = IMG + native_img
			for i in range(self.num_augmentations):
				shuffle(self.aug_seq)
				transform_list = self.create_transform_list(self.aug_seq)
				img_augmentated = Compose(transform_list)(img)
				IMG  = IMG + img_augmentated

		random.shuffle(IMG)
		ALLIMG_NP = np.stack(IMG, axis=0)
		OUT_IMAGE_NP = ALLIMG_NP[0:self.batch_size, :]
		return OUT_IMAGE_NP



def main():

	image_dir='./exampleImages'
	image_file_list = glob.glob(image_dir + '/*.png')
	output_size = (400, 400)
	transform_list = [RandFlip(), Rotate(20), NormalizeIntensity(), Rotate90()]

	#print(LoadImage(image_only=True)(image_file_list[0]).shape)
	#train_dataset=Dataset(image_file_list, transform_list, shuffle_transforms=0)
	#train_dataloader = DataLoader(train_dataset,  batch_size=4, num_workers=2)
	#for _batch_data in train_dataloader:
	#	img = _batch_data[0]

	image_loading_transforms = [LoadImage(image_only=True), AddChannel()]
	augmentation_dict = {'rotate': 3, 'flip': 2, 'rotate90': 1, 'intensityGaussian': 2, 'adjustContrast' : 2}

	img = AugmentData(image_loading_transforms=image_loading_transforms, augmentation_dict = augmentation_dict)(image_file_list)
	print(img.shape)


if __name__ == '__main__':
	main()
