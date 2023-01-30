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

import argparse
import json

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

site_names = ["site-1", "site-2", "site-3"]
merge_patients = True


def read_ground_truth(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    df = {"patient_id": [], "image": [], "label": [], "split": []}
    for split in data.keys():
        print(f"loading {split}: {len(data[split])} cases from {filename}")
        for item in data[split]:
            [df[k].append(item[k]) for k in item.keys()]
            df["split"].append(split)
            if "label" not in item.keys():
                df["label"].append(np.NAN)

    return pd.DataFrame(df)


def read_prediction(filename, gt, model_name):
    with open(filename, "r") as f:
        data = json.load(f)

    result = {}
    for s in site_names:
        result[s] = {
            "pred_probs": [],
            "gt_labels": [],
            "pred_probs_bin": [],
            "gt_labels_bin": [],
            "patient_ids": [],
        }
    for site in data.keys():
        for item in data[site][model_name]["test_probs"]:
            # multi-class
            assert len(item["probs"]) == 4, f"Expected four probs but got {len(item['probs'])}: {item['probs']}"
            result[site]["pred_probs"].append(item["probs"])
            gt_item = gt[gt["image"] == item["image"]]
            gt_label = gt_item["label"]
            assert len(gt_label) == 1, f"gt label was {gt_label}"
            result[site]["patient_ids"].append(gt_item["patient_id"].item())
            result[site]["gt_labels"].append(gt_label.item())

            # binary (non-dense vs dense)
            result[site]["pred_probs_bin"].append(np.sum(item["probs"][2::]))  # prob for dense (class 3 and 4).
            if gt_label.item() in [0, 1]:  # non-dense (class 1 and 2)
                result[site]["gt_labels_bin"].append(0)
            elif gt_label.item() in [2, 3]:  # dense (class 3 and 4)
                result[site]["gt_labels_bin"].append(1)
            else:
                raise ValueError(f"didn't expect a label of {gt_label}")
        assert (
            len(result[site]["gt_labels"])
            == len(result[site]["pred_probs"])
            == len(result[site]["gt_labels_bin"])
            == len(result[site]["pred_probs_bin"])
            == len(result[site]["patient_ids"])
        )
        assert len(np.unique(result[site]["gt_labels_bin"])) == 2, (
            f"Expected two kinds of binary labels but got " f"unique labels {np.unique(result[site]['gt_labels_bin'])}"
        )
    return result


def evaluate(site_result):
    gt_labels = site_result["gt_labels"]
    pred_probs = site_result["pred_probs"]
    gt_labels_bin = site_result["gt_labels_bin"]
    pred_probs_bin = site_result["pred_probs_bin"]

    # get pred labels
    pred_labels = []
    for prob in pred_probs:
        pred_labels.append(np.argmax(prob))

    assert len(gt_labels) == len(pred_labels) == len(gt_labels_bin) == len(pred_probs_bin)

    # multi-class metrics
    linear_kappa = sk_metrics.cohen_kappa_score(gt_labels, pred_labels, weights="linear")
    quadratic_kappa = sk_metrics.cohen_kappa_score(gt_labels, pred_labels, weights="quadratic")

    # per-image distance metrics
    dist = np.abs(np.squeeze(gt_labels) - np.squeeze(pred_labels))
    lin_dist = -dist
    quad_dist = -(dist**2)
    avg_lin_dist = np.mean(lin_dist)
    avg_quad_dist = np.mean(quad_dist)

    # binary metrics
    fpr, tpr, thresholds = sk_metrics.roc_curve(gt_labels_bin, pred_probs_bin, pos_label=1)
    auc = sk_metrics.auc(fpr, tpr)

    metrics = {
        "linear_kappa": linear_kappa,
        "quadratic_kappa": quadratic_kappa,
        "auc": auc,
        "lin_dist": lin_dist,
        "quad_dist": quad_dist,
        "avg_lin_dist": avg_lin_dist,
        "avg_quad_dist": avg_quad_dist,
    }
    print(
        f"evaluating {len(gt_labels)} predictions: "
        f"lin. kappa {linear_kappa:.3f}, "
        f"quad. kappa {quadratic_kappa:.3f}, "
        f"auc. {auc:.3f}, "
        f"avg. lin. dist {avg_lin_dist:.3f}, "
        f"avg. quad. dist {avg_quad_dist:.3f}, "
    )

    return metrics


def merge_patients(site_result):
    merged_results = {}
    for k in site_result.keys():
        merged_results[k] = []
        site_result[k] = np.array(site_result[k])  # needed for merging
    merged_results["counts"] = []

    patient_ids = site_result["patient_ids"]
    unique_patients = np.unique(patient_ids)
    print(f"Merging {len(patient_ids)} predictions from {len(unique_patients)} patients.")

    for patient in unique_patients:
        idx = np.where(patient_ids == patient)
        assert np.size(idx) > 0, "no matching patient found!"
        merged_results["patient_ids"].append(patient)
        merged_results["counts"].append(np.size(idx))
        # merge labels
        merged_results["gt_labels"].append(np.unique(site_result["gt_labels"][idx]))
        merged_results["gt_labels_bin"].append(np.unique(site_result["gt_labels_bin"][idx]))
        # merged labels should be all the same
        assert len(merged_results["gt_labels"][-1]) == 1
        assert len(merged_results["gt_labels_bin"][-1]) == 1
        # average probs
        merged_results["pred_probs"].append(np.mean(site_result["pred_probs"][idx], axis=0))
        merged_results["pred_probs_bin"].append(np.mean(site_result["pred_probs_bin"][idx]))
        assert len(merged_results["pred_probs"][-1]) == 4  # should be still four probs
        assert isinstance(merged_results["pred_probs_bin"][-1], float)  # should be just one prob
    print(f"Found patients with these nr of exams: {np.unique(merged_results['counts'])}")

    return merged_results


def compute_metrics(args):
    gt1 = read_ground_truth(args.gt1)
    gt2 = read_ground_truth(args.gt2)
    gt3 = read_ground_truth(args.gt3)
    ground_truth = pd.concat((gt1, gt2, gt3))
    pred_result = read_prediction(
        args.pred,
        gt=ground_truth[ground_truth["split"] == args.test_name],
        model_name=args.model_name,
    )  # read predictions and merge with ground truth

    print(f"Evaluating {args.model_name} on {args.test_name}:")
    overall_pred_result = {}

    metrics = {}
    for s in site_names:
        if merge_patients:
            pred_result[s] = merge_patients(pred_result[s])

        print(f"==={s}===")
        if not overall_pred_result:
            overall_pred_result = pred_result[s]
        else:
            [overall_pred_result[k].extend(pred_result[s][k]) for k in overall_pred_result.keys()]
        metrics[s] = evaluate(pred_result[s])
    print("===overall===")
    metrics["overall"] = evaluate(overall_pred_result)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt1", type=str, default="../../../dmist_files/dataset_site-1.json")
    parser.add_argument("--gt2", type=str, default="../../../dmist_files/dataset_site-2.json")
    parser.add_argument("--gt3", type=str, default="../../../dmist_files/dataset_site-3.json")
    parser.add_argument(
        "--pred",
        type=str,
        default="../../../results_acr_5-11-2022/result_server/predictions.json",
    )
    parser.add_argument("--test_name", type=str, default="test1")
    parser.add_argument("--model_name", type=str, default="SRV_best_FL_global_model.pt")
    args = parser.parse_args()

    metrics = compute_metrics(args)

    # print(f"Evaluation metrics for {args.model_name} on {args.test_name}:")
    # print(metrics)


if __name__ == "__main__":
    main()
