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

"""Mednist Shard Descriptor."""

import logging
from pathlib import Path
from typing import Dict
from typing import List

from monai.apps import MedNISTDataset

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor

logger = logging.getLogger(__name__)


class MedNistShardDataset(ShardDataset):
    """Mednist shard dataset class."""

    def __init__(
        self,
        data_items: List[Dict[str, str]],
        rank: int = 1,
        worldsize: int = 1,
    ) -> None:
        """Initialize Mednist Dataset."""
        self.rank = rank
        self.worldsize = worldsize
        self.data_items = data_items[self.rank - 1 :: self.worldsize]  # sharding

    def __len__(self) -> int:
        """Return the len of the shard dataset."""
        return len(self.data_items)

    def __getitem__(self, index: int) -> Dict[str, str]:
        """Return an item by the index."""
        return self.data_items[index]


class MedNistShardDescriptor(ShardDescriptor):
    """Shard descriptor class."""

    def __init__(self, data_folder: str = "./", rank_worldsize: str = "1,1", **kwargs) -> None:
        """Initialize MedNistShardDescriptor."""
        self.data_folder = data_folder
        self.rank, self.worldsize = map(int, rank_worldsize.split(","))
        self.download_data()

    def download_data(self) -> None:
        """Download prepared shard dataset."""
        dataset_train = MedNISTDataset(root_dir=self.data_folder, section="training", download=True, transform=None)
        dataset_val = MedNISTDataset(root_dir=self.data_folder, section="validation", download=True, transform=None)
        self.training_datadict = [
            {"fixed_hand": Path(item["image"]).absolute(), "moving_hand": Path(item["image"]).absolute()}
            for item in dataset_train.data
            if item["label"] == 4
        ]  # label 4 is for xray hands
        self.validation_datadict = [
            {"fixed_hand": Path(item["image"]).absolute(), "moving_hand": Path(item["image"]).absolute()}
            for item in dataset_val.data
            if item["label"] == 4
        ]  # label 4 is for xray hands

    def get_dataset(self, dataset_type: str) -> MedNistShardDataset:
        """Return a shard dataset by type."""
        if dataset_type == "train":
            return MedNistShardDataset(data_items=self.training_datadict, rank=self.rank, worldsize=self.worldsize)
        elif dataset_type == "validation":
            return MedNistShardDataset(data_items=self.validation_datadict, rank=self.rank, worldsize=self.worldsize)
        else:
            raise Exception("dataset_type should be one of ['train', 'validation']")

    @property
    def sample_shape(self) -> List[str]:
        """Return the sample shape info."""
        return ["64", "64", "1"]

    @property
    def target_shape(self) -> List[str]:
        """Return the target shape info."""
        return ["64", "64", "1"]

    @property
    def dataset_description(self) -> str:
        """Return the shard dataset description."""
        return f"Mednist dataset, shard number {self.rank}" f" out of {self.worldsize}"
