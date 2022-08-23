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

import monai.bundle
import torch
from monai.bundle import ConfigParser
from monai.engines import EnsembleEvaluator
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


class Const:
    CONFIGS = ("train.json", "train.yaml")
    MULTI_GPU_CONFIGS = ("multi_gpu_train.json", "multi_gpu_train.yaml")
    INFERENCE_CONFIGS = ("inference.json", "inference.yaml")
    METADATA_JSON = "metadata.json"

    KEY_DEVICE = "device"
    KEY_BUNDLE_ROOT = "bundle_root"
    KEY_NETWORK = "network"
    KEY_NETWORK_DEF = "network_def"
    KEY_DATASET_DIR = "dataset_dir"
    KEY_TRAIN_TRAINER_MAX_EPOCHS = "train#trainer#max_epochs"
    KEY_TRAIN_DATASET_DATA = "train#dataset#data"
    KEY_VALIDATE_DATASET_DATA = "validate#dataset#data"
    KEY_INFERENCE_DATASET_DATA = "dataset#data"
    KEY_MODEL_PYTORCH = "validate#handlers#-1#key_metric_filename"
    KEY_INFERENCE_POSTPROCESSING = "postprocessing"


class EnsembleTrainTask():
    def __init__(self, path):
        config_paths = [c for c in Const.CONFIGS if os.path.exists(os.path.join(path, "configs", c))]
        if not config_paths:
            logger.warning(f"Ignore {path} as there is no train config {Const.CONFIGS} exists")
            return

        self.bundle_path = path
        self.bundle_config_path = os.path.join(path, "configs", config_paths[0])

        self.bundle_config = ConfigParser()
        self.bundle_config.read_config(self.bundle_config_path)
        self.bundle_config.update({Const.KEY_BUNDLE_ROOT: self.bundle_path})

        self.bundle_metadata_path = os.path.join(path, "configs", Const.METADATA_JSON)

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

    def ensemble_inference(self, device, test_datalist, ensemble='Mean'):
        inference_config_paths = [c for c in Const.INFERENCE_CONFIGS if os.path.exists(os.path.join(self.bundle_path, "configs", c))]
        if not inference_config_paths:
            logger.warning(f"Ignore {self.bundle_path} as there is no inference config {Const.INFERENCE_CONFIGS} exists")
            return

        logger.info(f"Total Records in Test Dataset: {len(test_datalist)}")

        bundle_inference_config_path = os.path.join(self.bundle_path, "configs", inference_config_paths[0])
        bundle_inference_config = ConfigParser()
        bundle_inference_config.read_config(bundle_inference_config_path)
        bundle_inference_config.update({Const.KEY_BUNDLE_ROOT: self.bundle_path})
        bundle_inference_config.update({Const.KEY_INFERENCE_DATASET_DATA: test_datalist})

        # update postprocessing with mean ensemble or vote ensemble
        post_tranform = bundle_inference_config.config['postprocessing']
        ensemble_tranform = {
            "_target_": f"{ensemble}Ensembled",
            "keys": ["pred", "pred", "pred", "pred", "pred"],
            "output_key": "pred"
        }
        if ensemble == 'Mean':
            post_tranform["transforms"].insert(0, ensemble_tranform)
        elif ensemble == 'Vote':
            post_tranform["transforms"].insert(-1, ensemble_tranform)
        else:
            raise NotImplementedError
        bundle_inference_config.update({Const.KEY_INFERENCE_POSTPROCESSING: post_tranform})

        # update network weights
        _networks = [bundle_inference_config.get_parsed_content("network")]*5
        networks = []
        for i, _network in enumerate(_networks):
            _network.load_state_dict(torch.load(self.bundle_path+f"/models/model{i}.pt"))
            networks.append(_network)

        evaluator = EnsembleEvaluator(
            device=device,
            val_data_loader=bundle_inference_config.get_parsed_content("dataloader"),
            pred_keys=["pred", "pred", "pred", "pred", "pred"],
            networks=networks,
            inferer=bundle_inference_config.get_parsed_content("inferer"),
            postprocessing=bundle_inference_config.get_parsed_content("postprocessing"),
        )
        evaluator.run()
        logger.info(f"Inference Finished....")

    def __call__(self, request, datalist, test_datalist=None):
        dataset_dir = request.get("dataset_dir", None)
        if dataset_dir is None:
            logger.warning(f"Ignore dataset dir as there is no dataset dir exists")
            return

        train_ds, val_ds = self._partition_datalist(datalist, n_splits=request.get("n_splits", 5))
        fold = 0
        for _train_ds, _val_ds in zip(train_ds, val_ds):
            model_pytorch = f'model{fold}.pt'
            max_epochs = request.get("max_epochs", 50)
            multi_gpu = request.get("multi_gpu", False)
            multi_gpu = multi_gpu if torch.cuda.device_count() > 1 else False

            gpus = request.get("gpus", "all")
            gpus = list(range(torch.cuda.device_count())) if gpus == "all" else [int(g) for g in gpus.split(",")]
            logger.info(f"Using Multi GPU: {multi_gpu}; GPUS: {gpus}")
            logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

            device = self._device(request.get("device", "cuda:0"))
            logger.info(f"Using device: {device}")

            overrides = {
                Const.KEY_BUNDLE_ROOT: self.bundle_path,
                Const.KEY_TRAIN_TRAINER_MAX_EPOCHS: max_epochs,
                Const.KEY_TRAIN_DATASET_DATA: _train_ds,
                Const.KEY_VALIDATE_DATASET_DATA: _val_ds,
                Const.KEY_DATASET_DIR: dataset_dir,
                Const.KEY_MODEL_PYTORCH: model_pytorch,
            }

            if multi_gpu:
                pass
            else:
                train_config = ConfigParser()
                train_config.read_config(f=self.bundle_config.config)
                train_config.update(pairs=overrides)
                train_config_path = os.path.join(self.bundle_path, "configs", f"train_fold{fold}.json")
                ConfigParser.export_config_file(train_config.config, train_config_path, indent=2)
                monai.bundle.run(
                    "training",
                    meta_file=self.bundle_metadata_path,
                    config_file=train_config_path,
                )

            logger.info(f"Fold{fold} Training Finished....")

        if test_datalist is not None:
            device = self._device(request.get("device", "cuda:0"))
            self.ensemble_inference(device, test_datalist, ensemble=request.get("ensemble", "Mean"))


if __name__ == '__main__':
    request = {
        'dataset_dir': '/workspace/Data/Task09_Spleen',
        'max_epochs': 6,
        'ensemble': "Mean",
        'n_splits': 5
    }
    datalist_path = request['dataset_dir']+'/dataset.json'
    with open(datalist_path) as fp:
        datalist = json.load(fp)


    train_datalist = [{"image": d["image"].replace('./', f'{request["dataset_dir"]}/'), "label": d["label"].replace('./', f'{request["dataset_dir"]}/')} for d in datalist['training'] if d]
    test_datalist = [{"image": d.replace('./', f'{request["dataset_dir"]}/')} for d in datalist['test'] if d]
    bundle_root = '/workspace/Code/Bundles/spleen_ct_segmentation'
    EnsembleTrainTask = EnsembleTrainTask(bundle_root)
    EnsembleTrainTask(request, train_datalist, test_datalist)
