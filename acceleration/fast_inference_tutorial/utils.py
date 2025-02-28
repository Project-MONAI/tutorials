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


import os
import glob
import shutil
import monai
import pandas as pd
import numpy as np
from collections import OrderedDict
import torch
from ignite.engine import Engine
from ignite.engine import Events
from monai.engines import IterationEvents
from monai.bundle import trt_export
from monai.apps import download_and_extract


def prepare_test_datalist(root_dir):
    resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
    md5 = "410d4a301da4e5b2f6f86ec3ddba524e"

    compressed_file = os.path.join(root_dir, "Task09_Spleen.tar")
    data_root = os.path.join(root_dir, "Task09_Spleen")
    if not os.path.exists(data_root):
        download_and_extract(resource, compressed_file, root_dir, md5)

    nii_dir = os.path.join(data_root, "imagesTr_nii")
    if not os.path.exists(nii_dir):
        os.makedirs(nii_dir, exist_ok=True)
        train_gz_files = sorted(glob.glob(os.path.join(data_root, "imagesTr", "*.nii.gz")))
        for file in train_gz_files:
            new_file = file.replace(".nii.gz", ".nii")
            if not os.path.exists(new_file):
                os.system(f"gzip -dc {file} > {new_file}")
            shutil.copy(new_file, nii_dir)
    else:
        print(f"Test data already exists at {nii_dir}")

    train_files = sorted(glob.glob(os.path.join(nii_dir, "*.nii")))
    return train_files


def prepare_test_bundle(bundle_dir, bundle_name="spleen_ct_segmentation"):
    bundle_path = os.path.join(bundle_dir, bundle_name)
    if not os.path.exists(bundle_path):
        monai.bundle.download(name=bundle_name, bundle_dir=bundle_dir)
    else:
        print(f"Bundle already exists at {bundle_path}")
    return bundle_path


def prepare_tensorrt_model(bundle_path, trt_model_name="model_trt.ts"):
    output_path = os.path.join(bundle_path, "models", trt_model_name)
    if not os.path.exists(output_path):
        trt_export(
            net_id="network_def",
            filepath=output_path,
            ckpt_file=os.path.join(bundle_path, "models", "model.pt"),
            meta_file=os.path.join(bundle_path, "configs", "metadata.json"),
            config_file=os.path.join(bundle_path, "configs", "inference.json"),
            precision="fp16",
            dynamic_batchsize=[1, 4, 8],
            use_onnx=True,
            use_trace=True
        )
    else:
        print(f"TensorRT model already exists at {output_path}")


class CUDATimer:
    def __init__(self, type_str) -> None:
        self.time_list = []
        self.type_str = type_str

    def start(self) -> None:
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        self.starter.record()

    def end(self) -> None:
        self.ender.record()
        torch.cuda.synchronize()
        self.time_list.append(self.starter.elapsed_time(self.ender))

    def get_max(self) -> float:
        return max(self.time_list)

    def get_min(self) -> float:
        return min(self.time_list)

    def get_mean(self) -> float:
        np_time = np.array(self.time_list)
        return np.mean(np_time)

    def get_std(self) -> float:
        np_time = np.array(self.time_list)
        return np.std(np_time)

    def get_sum(self) -> float:
        np_time = np.array(self.time_list)
        return np.sum(np_time)

    def get_results_dict(self) -> OrderedDict:
        out_list = [
            ("total", self.get_sum()),
            ("min", self.get_min()),
            ("max", self.get_max()),
            ("mean", self.get_mean()),
            ("std", self.get_std()),
        ]
        return OrderedDict(out_list)


class TimerHandler:
    def __init__(self) -> None:
        self.run_timer = CUDATimer("RUN")
        self.epoch_timer = CUDATimer("EPOCH")
        self.iteration_timer = CUDATimer("ITERATION")
        self.net_forward_timer = CUDATimer("FORWARD")
        self.get_batch_timer = CUDATimer("PREPARE_BATCH")
        self.post_process_timer = CUDATimer("POST_PROCESS")
        self.timer_list = [
            self.run_timer,
            self.epoch_timer,
            self.iteration_timer,
            self.net_forward_timer,
            self.get_batch_timer,
            self.post_process_timer,
        ]

    def attach(self, engine: Engine) -> None:
        engine.add_event_handler(Events.STARTED, self.started, timer=self.run_timer)
        engine.add_event_handler(Events.EPOCH_STARTED, self.started, timer=self.epoch_timer)
        engine.add_event_handler(Events.ITERATION_STARTED, self.started, timer=self.iteration_timer)
        engine.add_event_handler(Events.GET_BATCH_STARTED, self.started, timer=self.get_batch_timer)
        engine.add_event_handler(Events.GET_BATCH_COMPLETED, self.completed, timer=self.get_batch_timer)
        engine.add_event_handler(Events.GET_BATCH_COMPLETED, self.started, timer=self.net_forward_timer)
        engine.add_event_handler(IterationEvents.FORWARD_COMPLETED, self.completed, timer=self.net_forward_timer)
        engine.add_event_handler(IterationEvents.FORWARD_COMPLETED, self.started, timer=self.post_process_timer)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, timer=self.post_process_timer)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, timer=self.iteration_timer)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed, timer=self.epoch_timer)
        engine.add_event_handler(Events.COMPLETED, self.completed, timer=self.run_timer)

    def started(self, engine: Engine, timer: CUDATimer) -> None:
        timer.start()

    def completed(self, engine: Engine, timer: CUDATimer) -> None:
        timer.end()

    def print_results(self) -> None:
        index = [x.type_str for x in self.timer_list]
        column_title = list(self.timer_list[0].get_results_dict().keys())
        column_title = [x + "/ms" for x in column_title]
        latency_list = [x for timer in self.timer_list for x in timer.get_results_dict().values()]
        latency_array = np.array(latency_list)
        latency_array = np.reshape(latency_array, (len(index), len(column_title)))
        df = pd.DataFrame(latency_array, index=index, columns=column_title)
        return df


def prepare_workflow(inference_config, meta_config, bundle_path, override):
    workflow = monai.bundle.ConfigWorkflow(
        workflow="infer",
        config_file=inference_config,
        meta_file=meta_config,
        logging_file=os.path.join(bundle_path, "configs", "logging.conf"),
        bundle_root=bundle_path,
        **override,
    )

    return workflow

def benchmark_workflow(workflow, timer, benchmark_type):
    workflow.initialize()
    timer.attach(workflow.evaluator)
    workflow.run()
    workflow.finalize()

    benchmark_df = timer.print_results()
    benchmark_df.to_csv(f"benchmark_{benchmark_type}.csv")

    return benchmark_df
