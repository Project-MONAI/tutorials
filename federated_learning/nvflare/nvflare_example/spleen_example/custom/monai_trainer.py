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

import torch.distributed as dist
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLConstants, ShareableKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.trainer import Trainer
from nvflare.common.signal import Signal
from nvflare.utils.fed_utils import generate_failure

from train_configer import TrainConfiger
from utils import (
    IterAggregateHandler,
    MONAIModelManager,
    TrainContext,
    get_lr_values,
    set_engine_state,
)


class MONAITrainer(Trainer):
    """
    This class implements a MONAI based trainer that can be used for Federated Learning.

    Args:
        aggregation_epochs: the number of training epochs for a round.
            This parameter only works when `aggregation_iters` is 0. Defaults to 1.
        aggregation_iters:  the number of training iterations for a round.
            If the value is larger than 0, the trainer will use iteration based aggregation
            rather than epoch based aggregation. Defaults to 0.

    """

    def __init__(self, aggregation_epochs: int = 1, aggregation_iters: int = 0):
        super().__init__()
        self.aggregation_epochs = aggregation_epochs
        self.aggregation_iters = aggregation_iters
        self.model_manager = MONAIModelManager()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _initialize_trainer(self, fl_ctx: FLContext):
        """
        The trainer's initialization function. At the beginning of a FL experiment,
        the train and evaluate engines, as well as train context and FL context
        should be initialized.
        """
        # Initialize train and evaluation engines.
        config_root = fl_ctx.get_prop(FLConstants.TRAIN_ROOT)
        fl_args = fl_ctx.get_prop(FLConstants.ARGS)

        conf = TrainConfiger(
            config_root=config_root,
            wf_config_file_name=fl_args.train_config,
            local_rank=fl_args.local_rank,
        )
        conf.configure()

        self.train_engine = conf.train_engine
        self.eval_engine = conf.eval_engine
        self.multi_gpu = conf.multi_gpu

        # for iterations based aggregation, the train engine should attach
        # the following handler.
        if self.aggregation_iters > 0:
            IterAggregateHandler(interval=self.aggregation_iters).attach(
                self.train_engine
            )

        # Instantiate a train context class. This instance is used to
        # save training related information such as current epochs, iterations
        # and the learning rate.
        self.train_ctx = TrainContext()
        self.train_ctx.initial_learning_rate = get_lr_values(
            self.train_engine.optimizer
        )

        # Initialize the FL context.
        fl_ctx.set_prop(FLConstants.MY_RANK, self.train_engine.state.rank)
        fl_ctx.set_prop(FLConstants.MODEL_NETWORK, self.train_engine.network)
        fl_ctx.set_prop(FLConstants.MULTI_GPU, self.multi_gpu)
        fl_ctx.set_prop(FLConstants.DEVICE, self.train_engine.state.device)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """
        This function is an extended function from the super class.
        It is used to perform the handler process based on the
        event_type. At the start point of a FL experiment, necessary
        components should be initialized. At the end of the experiment,
        the running engines should be terminated.

        Args:
            event_type: the type of event that will be fired. In MONAITrainer,
                only `START_RUN` and `END_RUN` need to be handled.
            fl_ctx: an `FLContext` object.

        """
        if event_type == EventType.START_RUN:
            self._initialize_trainer(fl_ctx)
        elif event_type == EventType.END_RUN:
            try:
                self.train_engine.terminate()
                self.eval_engine.terminate()
            except BaseException as e:
                self.logger.info(f"exception in closing engines {e}")

    def train(
        self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal
    ) -> Shareable:
        """
        This function is an extended function from the super class.
        As a supervised learning based trainer, the train function will run
        evaluate and train engines based on model weights from `shareable`.
        After fininshing training, a new `Shareable` object will be submitted
        to server for aggregation.

        Args:
            shareable: the `Shareable` object acheived from server.
            fl_ctx: the `FLContext` object achieved from server.
            abort_signal: if triggered, the training will be aborted.

        Returns:
            a new `Shareable` object to be submitted to server for aggregation.
        """
        # check abort signal
        self.logger.info(f"MonaiTrainer abort signal: {abort_signal.triggered}")
        if abort_signal.triggered:
            self.finalize(fl_ctx)
            shareable = generate_failure(fl_ctx=fl_ctx, reason="abort signal triggered")
            return shareable
        # achieve model weights
        if self.train_engine.state.rank == 0:
            model_weights = shareable[ShareableKey.MODEL_WEIGHTS]
            # load achieved model weights for the network (saved in fl_ctx)
            self.model_manager.assign_current_model(model_weights, fl_ctx)
        # for multi-gpu training, only rank 0 process will achieve the model weights.
        # Thus, it should be broadcasted to all processes.
        if self.multi_gpu:
            net = fl_ctx.get_prop(FLConstants.MODEL_NETWORK)
            for _, v in net.state_dict().items():
                dist.broadcast(v, src=0)

        # set engine state parameters, like number of training epochs/iterations.
        self.train_engine = set_engine_state(
            self.train_engine, self.aggregation_epochs, self.aggregation_iters
        )
        # get current epoch and iteration when a round starts
        self.train_ctx.epoch_of_start_time = self.train_engine.state.epoch
        self.train_ctx.iter_of_start_time = self.train_engine.state.iteration
        # execute validation at the beginning of every round
        self.eval_engine.run(self.train_engine.state.epoch + 1)
        self.train_ctx.fl_init_validation_metric = self.eval_engine.state.metrics.get(
            self.eval_engine.state.key_metric_name, -1
        )
        # record iteration and epoch data before training
        starting_iters = self.train_engine.state.iteration
        starting_epochs = self.train_engine.state.epoch
        self.train_engine.run()
        # calculate current iteration and epoch data after training
        self.train_ctx.current_iters = (
            self.train_engine.state.iteration - starting_iters
        )
        self.train_ctx.current_executed_epochs = (
            self.train_engine.state.epoch - starting_epochs
        )
        # create a new `Shareable` object
        if self.train_engine.state.rank == 0:
            self.train_ctx.set_context(self.train_engine, self.eval_engine)
            shareable = self.model_manager.generate_shareable(
                self.train_ctx,
                fl_ctx,
            )
        # update train context into FL context.
        fl_ctx.set_prop(FLConstants.TRAIN_CONTEXT, self.train_ctx)
        return shareable
