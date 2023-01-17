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

import csv


def writeCSV(filename, lines):
    with open(filename, "wb") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(lines)


def readCSV(filename):
    lines = []
    try:
        with open(filename, "rb") as f:
            csvreader = csv.reader(f)
            for line in csvreader:
                lines.append(line)
    except:
        with open(filename, "r") as f:
            csvreader = csv.reader(f)
            for line in csvreader:
                lines.append(line)
    return lines


def tryFloat(value):
    try:
        value = float(value)
    except:
        value = value

    return value


def getColumn(lines, columnid, elementType=""):
    column = []
    for line in lines:
        try:
            value = line[columnid]
        except:
            continue

        if elementType == "float":
            value = tryFloat(value)

        column.append(value)
    return column
