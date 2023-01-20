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

import ast
from types import SimpleNamespace
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from monai.transforms import LoadImage
from monai.transforms.transform import Transform
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: SimpleNamespace,
        aug: Transform,
        mode: str = "train",
    ):
        """

        Args:
            df: input dataframe.
            cfg: the config data, it must be based on `basic_cfg` in `default_config.py`.
            aug: transform(s) of the input data.
            mode: mode of the dataset. For `train` mode, annotations will also be loaded.

        """

        self.img_reader = LoadImage(image_only=True)
        self.cfg = cfg
        self.df = df.copy()
        self.annotated_df = pd.read_csv(cfg.data_dir + "train_annotations.csv")
        annotated_ids = self.annotated_df["StudyInstanceUID"].unique()
        self.is_annotated = (
            self.df["StudyInstanceUID"].isin(annotated_ids).astype(int).values
        )
        self.study_ids = self.df["StudyInstanceUID"].values
        self.annotated_df = self.annotated_df.groupby("StudyInstanceUID")
        self.label_cols = np.array(cfg.label_cols)

        if mode == "train":
            self.annot = pd.read_csv(cfg.data_dir + "train_annotations.csv")
            self.annot = self.annot[
                self.annot.StudyInstanceUID.isin(self.df.StudyInstanceUID)
            ]

        self.labels = self.df[self.label_cols].values
        self.mode = mode
        self.aug = aug
        self.data_folder = cfg.data_folder

    def get_thickness(self):
        """
        Get thickness from config data. The value determines the thichness of the polyline edges
        of the mask to be drawn.

        """
        if isinstance(self.cfg.thickness, list):
            thickness = np.random.randint(self.cfg.thickness[0], self.cfg.thickness[1])
        else:
            thickness = self.cfg.thickness

        return thickness

    def load_one(self, study_id: str):
        """
        Load image. The returned image has the shape (height, width, 1).

        """
        ext = self.cfg.image_extension
        fp = self.data_folder + study_id + ext
        img = self.img_reader(filename=fp).numpy().transpose(1, 0)
        img = img[:, :, None]

        return img

    def get_mask(self, study_id: str, img_shape: Tuple, is_annotated: int):
        """
        Get mask of image. The returned mask has the shape (height, width, 3),
        where 3 represents three different labels: ETT, NGT and CVC.

        """
        if is_annotated == 0:
            return np.zeros((img_shape[0], img_shape[1], self.cfg.seg_dim))

        df = self.annotated_df.get_group(study_id)
        mask = np.zeros((img_shape[0], img_shape[1], self.cfg.seg_dim))

        for idx, data in df.iterrows():

            xys = [
                np.array(ast.literal_eval(data["data"]))
                .clip(0, np.inf)
                .astype(np.int32)[:, None, :]
            ]

            m = np.zeros(img_shape)
            m = cv2.polylines(
                m, xys, False, 1, thickness=self.get_thickness(), lineType=cv2.LINE_AA
            )

            if self.cfg.seg_dim > 3:
                idx = np.where(self.label_cols == data["label"])[0][0]
            else:
                if "ETT" in data["label"] or self.cfg.seg_dim == 1:
                    idx = 0
                elif "NGT" in data["label"]:
                    idx = 1
                elif "CVC" in data["label"]:
                    idx = 2
                else:
                    continue

            mask[:, :, idx][:, :, None] = np.max(
                [mask[:, :, idx][:, :, None], m], axis=0
            )

        return mask

    def __getitem__(self, idx):

        study_id = self.study_ids[idx]
        label = self.labels[idx]
        is_annotated = self.is_annotated[idx]

        img = self.load_one(study_id)
        # convert the shape into (Channel, height, width)
        mask = self.get_mask(study_id, img.shape, is_annotated).transpose(2, 0, 1)
        data = {"input": torch.tensor(img.transpose(2, 0, 1)), "mask": torch.tensor(mask)}
        if self.aug:
            data = self.aug(data)

        return {
            "input": data["input"],
            "target": torch.tensor(label).float(),
            "mask": data["mask"],
            "is_annotated": torch.tensor(is_annotated).float(),
        }

    def __len__(self):
        return len(self.study_ids)
