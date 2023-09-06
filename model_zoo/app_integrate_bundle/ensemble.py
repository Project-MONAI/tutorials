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
import logging
import os
import argparse
import subprocess

import torch
from monai.transforms import Compose
from monai.transforms.post.dictionary import MeanEnsembled, VoteEnsembled
from monai.bundle import create_workflow
from monai.engines import EnsembleEvaluator
from monai.utils import optional_import

KFold, _ = optional_import("sklearn.model_selection", name="KFold")
logger = logging.getLogger(__name__)


class Const:
    CONFIGS = ("train.json", "train.yaml")
    MULTI_GPU_CONFIGS = ("multi_gpu_train.json", "multi_gpu_train.yaml")
    INFERENCE_CONFIGS = ("inference.json", "inference.yaml")
    LOGGING_CONFIG = "logging.conf"
    METADATA_JSON = "metadata.json"


class EnsembleTrainTask:
    """
    To construct an n-fold training and ensemble infer on any dataset.
    Just specify the bundle root path and data root path.
    Date root path also need a dataset.json which should be like:
        {
            "training": [
                {
                    "image": "./image1.nii.gz"
                    "label": "./label1.nii.gz"
                },
                {
                    "image": "./image2.nii.gz",
                    "label": "./label2.nii.gz"
                },
                ...
            ],
            "test": [
                {
                    "image": "./image.nii.gz"
                },
                ...
            ]
        }

    Args:
        path: bundle root path where your place the download bundle
    """

    def __init__(self, path):
        config_paths = [c for c in Const.CONFIGS if os.path.exists(os.path.join(path, "configs", c))]
        if not config_paths:
            logger.warning(f"Ignore {path} as there is no train config {Const.CONFIGS} exists")
            return

        self.bundle_path = path
        self.bundle_config_path = os.path.join(path, "configs", config_paths[0])
        self.bundle_metadata_path = os.path.join(path, "configs", Const.METADATA_JSON)
        self.bundle_logging_path = os.path.join(path, "configs", Const.LOGGING_CONFIG)

        self.train_workflow = create_workflow(config_file=self.bundle_config_path, workflow_type="train")

    def _partition_datalist(self, datalist, n_splits=5, shuffle=False):
        logger.info(f"Total Records in Dataset: {len(datalist)}")
        kfold = KFold(n_splits=n_splits, shuffle=shuffle)

        train_datalist, val_datalist = [], []
        for train_idx, valid_idx in kfold.split(datalist):
            train_datalist.append([datalist[i] for i in train_idx])
            val_datalist.append([datalist[i] for i in valid_idx])

        logger.info(f"Total Records for Training: {len(train_datalist[0])}")
        logger.info(f"Total Records for Validation: {len(val_datalist[0])}")
        return train_datalist, val_datalist

    def _device(self, str):
        return torch.device(str if torch.cuda.is_available() else "cpu")

    def ensemble_inference(self, device, test_datalist, ensemble="Mean"):
        inference_config_paths = [
            c for c in Const.INFERENCE_CONFIGS if os.path.exists(os.path.join(self.bundle_path, "configs", c))
        ]
        if not inference_config_paths:
            logger.warning(
                f"Ignore {self.bundle_path} as there is no inference config {Const.INFERENCE_CONFIGS} exists"
            )
            return

        logger.info(f"Total Records in Test Dataset: {len(test_datalist)}")

        bundle_inference_config_path = os.path.join(self.bundle_path, "configs", inference_config_paths[0])
        inference_workflow = create_workflow(config_file=bundle_inference_config_path, workflow_type="inference")
        inference_workflow.dataset_data = test_datalist
        # this application has an additional requirement for the bundle workflow to provide the property dataloader
        inference_workflow.add_property(name="dataloader", required=True, config_id="dataloader")
        inference_workflow.initialize()

        # update postprocessing with mean ensemble or vote ensemble
        _ensemble_transform = MeanEnsembled if ensemble == "Mean" else VoteEnsembled
        ensemble_transform = _ensemble_transform(keys=["pred"] * args.n_splits, output_key="pred")
        if ensemble == "Mean":
            _postprocessing = Compose((ensemble_transform, inference_workflow.postprocessing))
        elif ensemble == "Vote":
            _postprocessing = Compose((inference_workflow.postprocessing, ensemble_transform))
        else:
            raise NotImplementedError

        # update network weights
        networks = []
        for i in range(args.n_splits):
            _network = inference_workflow.network_def.to(device)
            _network.load_state_dict(torch.load(self.bundle_path + f"/models/model_fold{i}.pt", map_location=device))
            networks.append(_network)

        inference_workflow.evaluator = EnsembleEvaluator(
            device=device,
            val_data_loader=inference_workflow.dataloader,
            pred_keys=["pred"] * args.n_splits,
            networks=networks,
            inferer=inference_workflow.inferer,
            postprocessing=_postprocessing,
        )
        inference_workflow.run()
        inference_workflow.finalize()
        logger.info("Inference Finished....")

    def __call__(self, args, datalist, test_datalist=None):
        dataset_dir = args.dataset_dir
        if dataset_dir is None:
            logger.warning("Ignore dataset dir as there is no dataset dir exists")
            return

        train_ds, val_ds = self._partition_datalist(datalist[:7], n_splits=args.n_splits)
        fold = 0
        for _train_ds, _val_ds in zip(train_ds, val_ds):
            max_epochs = args.epochs
            multi_gpu = args.multi_gpu
            multi_gpu = multi_gpu if torch.cuda.device_count() > 1 else False

            gpus = args.gpus
            gpus = list(range(torch.cuda.device_count())) if gpus == "all" else [int(g) for g in gpus.split(",")]
            logger.info(f"Using Multi GPU: {multi_gpu}; GPUS: {gpus}")
            logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

            device = self._device(args.device)
            logger.info(f"Using device: {device}")

            if multi_gpu:
                train_datalist_path = os.path.join(self.bundle_path, "configs", f"train_datalist_fold{fold}.json")
                val_datalist_path = os.path.join(self.bundle_path, "configs", f"val_datalist_fold{fold}.json")
                with open(train_datalist_path, "w") as f:
                    json.dump(_train_ds, f)
                with open(val_datalist_path, "w") as f:
                    json.dump(_val_ds, f)

                config_paths = [
                    c for c in Const.MULTI_GPU_CONFIGS if os.path.exists(os.path.join(self.bundle_path, "configs", c))
                ]
                if not config_paths:
                    logger.warning(
                        f"Ignore Multi-GPU Training; No multi-gpu train config {Const.MULTI_GPU_CONFIGS} exists"
                    )
                    return

                multi_gpu_train_path = os.path.join(self.bundle_path, "configs", config_paths[0])

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ",".join([str(g) for g in gpus])
                logger.info(f"Using CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}")
                cmd = [
                    "torchrun",
                    "--standalone",
                    "--nnodes=1",
                    f"--nproc_per_node={len(gpus)}",
                    "-m",
                    "monai.bundle",
                    "run",
                    "--meta_file",
                    self.bundle_metadata_path,
                    "--config_file",
                    f"['{self.bundle_config_path}','{multi_gpu_train_path}']",
                    "--logging_file",
                    self.bundle_logging_path,
                    "--bundle_root",
                    self.bundle_path,
                    "--epochs",
                    str(max_epochs),
                    "--dataset_dir",
                    dataset_dir,
                    "--train#dataset#data",
                    f"%{train_datalist_path}",
                    "--validate#dataset#data",
                    f"%{val_datalist_path}",
                ]
                self.run_command(cmd, env)
            else:
                self.train_workflow.bundle_root = self.bundle_path
                self.train_workflow.max_epochs = max_epochs
                self.train_workflow.train_dataset_data = _train_ds
                self.train_workflow.val_dataset_data = _val_ds
                self.train_workflow.dataset_dir = dataset_dir
                self.train_workflow.device = device

                self.train_workflow.initialize()
                self.train_workflow.run()
                self.train_workflow.finalize()

            _model_path = f"{self.bundle_path}/models/model.pt"
            os.rename(_model_path, f"{self.bundle_path}/models/model_fold{fold}.pt")
            logger.info(f"Fold {fold} Training Finished....")
            fold += 1

        if test_datalist is not None:
            device = self._device(args.device)
            self.ensemble_inference(device, test_datalist, ensemble=args.ensemble)

    def run_command(self, cmd, env):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True, env=env)
        while process.poll() is None:
            line = process.stdout.readline()
            line = line.rstrip()
            if line:
                print(line, flush=True)

        logger.info(f"Return code: {process.returncode}")
        process.stdout.close()


if __name__ == "__main__":
    """
    Usage
        first download a bundle to somewhere as your bundle_root path
        split your data into train and test datalist
            train datalist: [
                {
                    "image": $image1_path
                    "label": $label1_path
                },
                {
                    "image": $image2_path,
                    "label": $label2_path
                },
                ...
            ]
            test_datalist: [
                {
                    "image": $image1_path
                },
                ...
            ]
        python easy_integrate_bundle.py --bundle_root $bundle_root_path --dataset_dir $data_root_path
    """
    parser = argparse.ArgumentParser(description="Run an ensemble train task using bundle.")

    parser.add_argument("--ensemble", default="Mean", choices=["Mean", "Vote"], type=str, help="way of ensemble")
    parser.add_argument("--bundle_root", default="", type=str, help="root bundle dir")
    parser.add_argument("--dataset_dir", default="", type=str, help="root data dir")
    parser.add_argument("--epochs", default=6, type=int, help="max epochs")
    parser.add_argument("--n_splits", default=5, type=int, help="n fold split")
    parser.add_argument("--multi_gpu", default=False, type=bool, help="whether use multigpu")
    parser.add_argument("--device", default="cuda", type=str, help="device")
    parser.add_argument("--gpus", default="all", type=str, help="which gpu to use")

    args = parser.parse_args()
    gpus = list(range(torch.cuda.device_count())) if args.gpus == "all" else [int(g) for g in args.gpus.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpus)
    datalist_path = args.dataset_dir + "/dataset.json"
    with open(datalist_path) as fp:
        datalist = json.load(fp)

    train_datalist = [
        {
            "image": d["image"].replace("./", f"{args.dataset_dir}/"),
            "label": d["label"].replace("./", f"{args.dataset_dir}/"),
        }
        for d in datalist["training"]
        if d
    ]
    test_datalist = [{"image": d.replace("./", f"{args.dataset_dir}/")} for d in datalist["test"] if d]
    traintask = EnsembleTrainTask(args.bundle_root)
    traintask(args, train_datalist, test_datalist)
