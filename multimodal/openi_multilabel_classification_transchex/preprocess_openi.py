# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from os import listdir
from os.path import isfile, join
import numpy as np
from xml.dom import minidom
from PIL import Image
import pandas as pd
import xml.etree.ElementTree as ET

def create_report(img_names_list_, report_list_, gt_list_, save_add):
    pd.DataFrame({'id': img_names_list_, 'report': report_list_, 'Atelectasis': gt_list_[:, 0],
                  'Cardiomegaly': gt_list_[:, 1], 'Consolidation': gt_list_[:, 2],'Edema': gt_list_[:, 3],
                  'Enlarged-Cardiomediastinum': gt_list_[:, 4], 'Fracture': gt_list_[:, 5], 'Lung-Lesion': gt_list_[:, 6],
                  'Lung-Opacity': gt_list_[:, 7], 'No-Finding': gt_list_[:, 8], 'Pleural-Effusion': gt_list_[:, 9],
                  'Pleural_Other': gt_list_[:, 10], 'Pneumonia': gt_list_[:, 11], 'Pneumothorax': gt_list_[:, 12],
                  'Support-Devices': gt_list_[:, 13]}).to_csv(save_add, index=False)

report_file_add= './monai_data/dataset_orig/NLMCXR_reports/ecgen-radiology'
img_file_add= './monai_data/dataset_orig/NLMCXR_png'
npy_add= './monai_data/TransChex_openi/'

img_save_add = './monai_data/dataset_proc/images'
report_train_save_add = './monai_data/dataset_proc/train.csv'
report_val_save_add = './monai_data/dataset_proc/validation.csv'
report_test_save_add = './monai_data/dataset_proc/test.csv'

if not os.path.isdir(img_save_add):
    os.makedirs(img_save_add)
report_files = [f for f in listdir(report_file_add) if isfile(join(report_file_add, f))]

train_data = np.load(npy_add+'train.npy', allow_pickle=True).item()
train_data_id = train_data['id_GT']
train_data_gt = train_data['label_GT']

val_data = np.load(npy_add+'validation.npy', allow_pickle=True).item()
val_data_id = val_data['id_GT']
val_data_gt = val_data['label_GT']

test_data = np.load(npy_add+'test.npy', allow_pickle=True).item()
test_data_id = test_data['id_GT']
test_data_gt = test_data['label_GT']

all_cases = np.union1d(np.union1d(train_data_id, val_data_id), test_data_id)

img_names_list_train = []
img_names_list_val = []
img_names_list_test = []

report_list_train = []
report_list_val = []
report_list_test = []

gt_list_train = []
gt_list_val = []
gt_list_test = []

for file in report_files:
    print('Processing {}'.format(file))
    add_xml = os.path.join(report_file_add, file)
    docs = minidom.parse(add_xml)
    tree = ET.parse(add_xml)
    for node in tree.iter('AbstractText'):
        i = 0
        for elem in node.iter():
            if elem.attrib['Label'] == "FINDINGS":
                if elem.text == None:
                    report = "FINDINGS : "
                else:
                    report = "FINDINGS : " + elem.text
            elif elem.attrib['Label'] == "IMPRESSION":
                if elem.text == None:
                    report = report + " IMPRESSION : "
                else:
                    report = report + " IMPRESSION : " + elem.text
    images = docs.getElementsByTagName("parentImage")
    for i in images:
        img_name = i.getAttribute("id") + '.png'
        if img_name in all_cases:
            Image.open(os.path.join(img_file_add, img_name)).resize((512, 512)).save(
                os.path.join(img_save_add, img_name))
            if img_name in train_data_id:
                img_names_list_train.append(img_name)
                report_list_train.append(report)
                gt_list_train.append(train_data_gt[np.where(train_data_id==img_name)[0][0]])
            elif img_name in val_data_id:
                img_names_list_val.append(img_name)
                report_list_val.append(report)
                gt_list_val.append(val_data_gt[np.where(val_data_id == img_name)[0][0]])
            elif img_name in test_data_id:
                img_names_list_test.append(img_name)
                report_list_test.append(report)
                gt_list_test.append(test_data_gt[np.where(test_data_id == img_name)[0][0]])

datasets = [{"save_add": report_train_save_add,
              "img_name": np.array(img_names_list_train),
              "report": np.array(report_list_train),
              "gt": np.array(gt_list_train)},
            {"save_add": report_val_save_add,
             "img_name": np.array(img_names_list_val),
             "report": np.array(report_list_val),
             "gt": np.array(gt_list_val)},
            {"save_add": report_test_save_add,
             "img_name": np.array(img_names_list_test),
             "report": np.array(report_list_test),
             "gt": np.array(gt_list_test)}
            ]
for dataset in datasets:
    create_report(dataset["img_name"], dataset["report"], dataset["gt"], dataset["save_add"])

print('Processed Dataset Files Are Saved !')
