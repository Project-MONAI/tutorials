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

import json
import os


def strip_and_split(dataset_filename, strip_set):
    with open(dataset_filename, "r") as f:
        data = json.load(f)

    # remove labels
    [x.pop("label") for x in data[strip_set]]
    new_data = {
        "train": data["train"],  # keep the same train set in both cases
        "test": data[strip_set],
    }
    print(f"removed {len(data[strip_set])} labels from `{strip_set}`")
    return new_data


def main():
    datalist_rootdir = "../../../data"
    for client_id in ["site-1", "site-2", "site-3"]:
        print(f"processing {client_id}")
        new_datalist1 = strip_and_split(
            os.path.join(datalist_rootdir, f"./dataset_{client_id}.json"),
            strip_set="test1",
        )
        new_datalist2 = strip_and_split(
            os.path.join(datalist_rootdir, f"./dataset_{client_id}.json"),
            strip_set="test2",
        )
        with open(
            os.path.join(
                datalist_rootdir, f"./dataset_blinded_{client_id}.json"
            ),
            "w",
        ) as f:
            json.dump(new_datalist1, f, indent=4)
        with open(
            os.path.join(
                datalist_rootdir, f"./dataset_blinded_phase2_{client_id}.json"
            ),
            "w",
        ) as f:
            json.dump(new_datalist2, f, indent=4)


if __name__ == "__main__":
    main()
