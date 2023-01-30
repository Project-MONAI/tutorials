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

import json
import csv
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        default=[],
        help="input json",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output csv",
    )

    args = parser.parse_args()

    in_json_list = args.input
    out_csv = args.output

    with open(out_csv, "w", newline="") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(["seriesuid", "coordX", "coordY", "coordZ", "probability"])
        for in_json in in_json_list:
            result = json.load(open(in_json, "r"))
            for subj in result["validation"]:
                seriesuid = os.path.split(subj["image"])[-1][:-7]
                for b in range(len(subj["box"])):
                    spamwriter.writerow([seriesuid] + subj["box"][b][0:3] + [subj["score"][b]])


if __name__ == "__main__":
    main()
