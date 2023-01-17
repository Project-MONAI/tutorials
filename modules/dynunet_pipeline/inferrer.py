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
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from ignite.engine import Engine
from monai.data import decollate_batch
from monai.data.nifti_writer import write_nifti
from monai.engines import SupervisedEvaluator
from monai.engines.utils import IterationEvents
from monai.inferers import Inferer
from monai.networks.utils import eval_mode
from monai.transforms import AsDiscrete
from torch.utils.data import DataLoader

from transforms import recovery_prediction


class DynUNetInferrer(SupervisedEvaluator):
    """
    This class inherits from SupervisedEvaluator in MONAI, and is used with DynUNet
    on Decathlon datasets. As a customized inferrer, some of the arguments from
    SupervisedEvaluator are not supported. For example, the actual
    post processing method used is hard coded in the `_iteration` function, thus the
    argument `postprocessing` from SupervisedEvaluator is not exist. If you need
    to change the post processing way, please modify the `_iteration` function directly.

    Args:
        device: an object representing the device on which to run.
        val_data_loader: Ignite engine use data_loader to run, must be
            torch.DataLoader.
        network: use the network to run model forward.
        output_dir: the path to save inferred outputs.
        num_classes: the number of classes (output channels) for the task.
        inferer: inference method that execute model forward on input data, like: SlidingWindow, etc.
        amp: whether to enable auto-mixed-precision evaluation, default is False.
        tta_val: whether to do the 8 flips (8 = 2 ** 3, where 3 represents the three dimensions)
            test time augmentation, default is False.

    """

    def __init__(
        self,
        device: torch.device,
        val_data_loader: DataLoader,
        network: torch.nn.Module,
        output_dir: str,
        num_classes: Union[str, int],
        inferer: Optional[Inferer] = None,
        amp: bool = False,
        tta_val: bool = False,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            network=network,
            inferer=inferer,
            amp=amp,
        )

        if not isinstance(num_classes, int):
            num_classes = int(num_classes)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
        self.output_dir = output_dir
        self.tta_val = tta_val
        self.num_classes = num_classes

    def _iteration(
        self, engine: Engine, batchdata: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        callback function for the Supervised Evaluation processing logic of 1 iteration in Ignite Engine.
        Return below item in a dictionary:
            - PRED: prediction result of model.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        batch = self.prepare_batch(batchdata, engine.state.device, engine.non_blocking)
        if len(batch) == 2:
            inputs, _ = batch
            args: Tuple = ()
            kwargs: Dict = {}
        else:
            inputs, _, args, kwargs = batch

        def _compute_pred():
            ct = 1.0
            pred = self.inferer(inputs, self.network, *args, **kwargs).cpu()
            pred = nn.functional.softmax(pred, dim=1)
            if not self.tta_val:
                return pred
            else:
                for dims in [[2], [3], [4], (2, 3), (2, 4), (3, 4), (2, 3, 4)]:
                    flip_inputs = torch.flip(inputs, dims=dims)
                    flip_pred = torch.flip(
                        self.inferer(flip_inputs, self.network).cpu(), dims=dims
                    )
                    flip_pred = nn.functional.softmax(flip_pred, dim=1)
                    del flip_inputs
                    pred += flip_pred
                    del flip_pred
                    ct += 1
                return pred / ct

        # execute forward computation
        with eval_mode(self.network):
            if self.amp:
                with torch.cuda.amp.autocast():
                    predictions = _compute_pred()
            else:
                predictions = _compute_pred()

        inputs = inputs.cpu()
        predictions = self.post_pred(decollate_batch(predictions)[0])

        affine = batchdata["image_meta_dict"]["affine"].numpy()[0]
        resample_flag = batchdata["resample_flag"]
        anisotrophy_flag = batchdata["anisotrophy_flag"]
        crop_shape = batchdata["crop_shape"][0].tolist()
        original_shape = batchdata["original_shape"][0].tolist()

        if resample_flag:
            # convert the prediction back to the original (after cropped) shape
            predictions = recovery_prediction(
                predictions.numpy(), [self.num_classes, *crop_shape], anisotrophy_flag
            )
        else:
            predictions = predictions.numpy()

        predictions = np.argmax(predictions, axis=0)

        # pad the prediction back to the original shape
        predictions_org = np.zeros([*original_shape])
        box_start, box_end = batchdata["bbox"][0]
        h_start, w_start, d_start = box_start
        h_end, w_end, d_end = box_end
        predictions_org[h_start:h_end, w_start:w_end, d_start:d_end] = predictions
        del predictions

        filename = batchdata["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]

        print(
            "save {} with shape: {}, mean values: {}".format(
                filename, predictions_org.shape, predictions_org.mean()
            )
        )
        write_nifti(
            data=predictions_org,
            file_name=os.path.join(self.output_dir, filename),
            affine=affine,
            resample=False,
            output_dtype=np.uint8,
        )
        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        return {"pred": predictions_org}
