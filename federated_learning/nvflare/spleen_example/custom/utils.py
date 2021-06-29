# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from typing import Dict

import numpy as np
import torch
from ignite.engine import Engine, Events
from nvflare.apis.fl_constant import FLConstants, ShareableKey, ShareableValue
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from torch.optim import Optimizer


class TrainContext:
    """
    Train Context class contains training related parameters/variables,
    such as learning rate, number of gpus and current training iterations.
    """

    def __init__(self):
        self.initial_learning_rate = 0
        self.current_learning_rate = 0
        self.current_iters = 0
        self.current_executed_epochs = 0
        self.fl_init_validation_metric = 0
        self.epoch_of_start_time = 0
        self.iter_of_start_time = 0

    def set_context(self, train_engine: Engine, eval_engine: Engine):
        """
        This function is usually called after train engine has finished running.
        The variables that updated here will add to the shareable object and then
        submit to server. You can add other variables in this function if they are
        needed to be shared.
        """
        self.current_learning_rate = get_lr_values(train_engine.optimizer)


class MONAIModelManager:
    def __init__(self):
        self.logger = logging.getLogger("ModelShareableManager")

    def assign_current_model(
        self, model_weights: Dict[str, np.ndarray], fl_ctx: FLContext
    ):
        """
        This function is used to load provided weights for the network saved
        in FL context.
        """
        net = fl_ctx.get_prop(FLConstants.MODEL_NETWORK)
        if fl_ctx.get_prop(FLConstants.MULTI_GPU):
            net = net.module

        local_var_dict = net.state_dict()
        model_keys = model_weights.keys()
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = model_weights[var_name]
                try:
                    local_var_dict[var_name] = torch.as_tensor(weights)
                except Exception as e:
                    raise ValueError(
                        "Convert weight from {} failed with error: {}".format(
                            var_name, str(e)
                        )
                    )

        net.load_state_dict(local_var_dict)

    def extract_model(self, fl_ctx: FLContext) -> Dict[str, np.ndarray]:
        """
        This function is used to extract weights of the network saved in FL
        context.
        The extracted weights will be converted into a numpy array based dict.
        """
        net = fl_ctx.get_prop(FLConstants.MODEL_NETWORK)
        if fl_ctx.get_prop(FLConstants.MULTI_GPU):
            net = net.module
        local_state_dict = net.state_dict()
        local_model_dict = {}
        for var_name in local_state_dict:
            try:
                local_model_dict[var_name] = local_state_dict[var_name].cpu().numpy()
            except Exception as e:
                raise ValueError(
                    "Convert weight from {} failed with error: {}".format(
                        var_name, str(e)
                    )
                )

        return local_model_dict

    def generate_shareable(self, train_ctx: TrainContext, fl_ctx: FLContext):
        """
        This function is used to generate a shareable instance according to
        the train context and FL context.
        A Shareable instance can not only contain model weights, but also
        some additional information that clients want to share. These information
        should be added into ShareableKey.META.
        """

        # input the initlal metric into meta data. You can also add other parameters.
        meta_data = {}
        meta_data[FLConstants.INITIAL_METRICS] = train_ctx.fl_init_validation_metric
        meta_data[FLConstants.CURRENT_LEARNING_RATE] = train_ctx.current_learning_rate

        shareable = Shareable()
        shareable[ShareableKey.TYPE] = ShareableValue.TYPE_WEIGHT_DIFF
        shareable[ShareableKey.DATA_TYPE] = ShareableValue.DATA_TYPE_UNENCRYPTED
        shareable[ShareableKey.MODEL_WEIGHTS] = self.extract_model(fl_ctx)
        shareable[ShareableKey.META] = meta_data

        return shareable


class IterAggregateHandler:
    """
    This class implements an event handler for iteration based aggregation.
    """

    def __init__(self, interval: int):
        self.interval = interval

    def attach(self, engine: Engine):
        engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.interval), self)

    def __call__(self, engine: Engine):
        engine.terminate()
        # save current iteration for next round
        engine.state.dataloader_iter = engine._dataloader_iter
        if engine.state.iteration % engine.state.epoch_length == 0:
            # if current iteration is end of 1 epoch, manually trigger epoch completed event
            engine._fire_event(Events.EPOCH_COMPLETED)


def get_lr_values(optimizer: Optimizer):
    """
    This function is used to get the learning rates of the optimizer.
    """
    return [group["lr"] for group in optimizer.state_dict()["param_groups"]]


def set_engine_state(engine: Engine, aggregation_epochs: int, aggregation_iters: int):
    """
    This function is used to set the engine's state parameters according to
    the aggregation ways (iteration based or epoch based).

    Args:
        engine: the engine that to be processed.
        aggregation_epochs: the number of epochs before aggregation.
            This parameter only works when `aggregation_iters` is 0.
        aggregation_iters:  the number of iterations before aggregation.
            If the value is larger than 0, the engine will use iteration based aggregation
            rather than epoch based aggregation.

    """
    if aggregation_iters > 0:
        next_aggr_iter = engine.state.iteration + aggregation_iters
        engine.state.max_epochs = math.ceil(next_aggr_iter / engine.state.epoch_length)
        previous_iter = engine.state.iteration % engine.state.epoch_length
        if engine.state.iteration > 0 and previous_iter != 0:
            # init to continue from previous epoch
            engine.state.epoch -= 1
            if hasattr(engine.state, "dataloader_iter"):
                # initialize to continue from previous iteration
                engine._init_iter.append(previous_iter)
                engine._dataloader_iter = engine.state.dataloader_iter
    else:
        engine.state.max_epochs = engine.state.epoch + aggregation_epochs

    return engine
