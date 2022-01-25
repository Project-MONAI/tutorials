#!/usr/bin/env python

import csv
import json
import pickle


# load annotation.csv
annotation_filename = "/home/dongy/Data/LUNA16/NGC/annotations.csv"
with open(annotation_filename) as csv_file:
    csv_data = csv.reader(csv_file)
    rows = []
    for row in csv_data:
        rows.append(row)

box_data = {}
for _j in range(1, len(rows)):
    if rows[_j][0] not in box_data.keys():
        box_data[rows[_j][0]] = []

    box_data[rows[_j][0]].append(
        [
            float(rows[_j][1]),
            float(rows[_j][2]),
            float(rows[_j][3]),
            float(rows[_j][4]),
        ]
    )

# load splits.pkl
pkl_data = pickle.load(open("splits.pkl", "rb"))

for _i in range(len(pkl_data)):
    json_data = {}
    json_data["training"] = []
    json_data["validation"] = []

    for _j in range(len(pkl_data[_i]["train"])):
        data_point = {}
        data_point["image"] = pkl_data[_i]["train"][_j] + ".mhd"
        data_point["label"] = []
        data_point["box"] = []

        key = pkl_data[_i]["train"][_j]
        key = key.replace("_", ".")
        print("key:", key)

        if key in box_data.keys():
            for _k in range(len(box_data[key])):
                data_point["label"].extend([0])
                box = box_data[key][_k]
                data_point["box"].append(
                    [
                        box[0] - 0.5 * box[3],
                        box[0] + 0.5 * box[3],
                        box[1] - 0.5 * box[3],
                        box[1] + 0.5 * box[3],
                        box[2] - 0.5 * box[3],
                        box[2] + 0.5 * box[3],
                    ]
                )

            json_data["training"].append(data_point)

    for _j in range(len(pkl_data[_i]["val"])):
        data_point = {}
        data_point["image"] = pkl_data[_i]["val"][_j] + ".mhd"
        data_point["label"] = []
        data_point["box"] = []

        key = pkl_data[_i]["val"][_j]
        key = key.replace("_", ".")
        print("key:", key)

        if key in box_data.keys():
            for _k in range(len(box_data[key])):
                data_point["label"].extend([0])
                box = box_data[key][_k]
                data_point["box"].append(
                    [
                        box[0] - 0.5 * box[3],
                        box[0] + 0.5 * box[3],
                        box[1] - 0.5 * box[3],
                        box[1] + 0.5 * box[3],
                        box[2] - 0.5 * box[3],
                        box[2] + 0.5 * box[3],
                    ]
                )

            json_data["validation"].append(data_point)

    with open("dataset_fold" + str(_i) + ".json", "w") as outfile:
        json.dump(json_data, outfile, sort_keys=True, indent=4, ensure_ascii=False)
