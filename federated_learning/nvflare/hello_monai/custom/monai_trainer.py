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


from typing import Dict

import numpy as np
import torch
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants

from train_configer import TrainConfiger


class MONAITrainer(Executor):
    """
    This class implements a MONAI based trainer that can be used for Federated Learning with NVFLARE.

    Args:
        aggregation_epochs: the number of training epochs for a round. Defaults to 1.

    """

    def __init__(
        self,
        aggregation_epochs: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
    ):
        super().__init__()
        """
        Trainer init happens at the very beginning, only the basic info regarding the trainer is set here,
        and the actual run has not started at this point.
        """
        self.aggregation_epochs = aggregation_epochs
        self._train_task_name = train_task_name

    def _initialize_trainer(self, fl_ctx: FLContext):
        """
        The trainer's initialization function. At the beginning of a FL experiment,
        the train and evaluate engines, as well as train context and FL context
        should be initialized.
        """
        # Initialize train and evaluation engines.
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        # will update multi-gpu supports later
        # num_gpus = fl_ctx.get_prop(AppConstants.NUMBER_OF_GPUS, 1)
        # self.multi_gpu = num_gpus > 1
        self.client_name = fl_ctx.get_identity_name()
        self.log_info(
            fl_ctx,
            f"Client {self.client_name} initialized at \n {app_root} \n with args: {fl_args}",
        )
        conf = TrainConfiger(
            app_root=app_root,
            wf_config_file_name=fl_args.train_config,
            local_rank=fl_args.local_rank,
        )
        conf.configure()

        # train_engine, and eval_engine are MONAI engines that will be used for training and validation.
        # The corresponding training/validation settings, such as transforms, network and dataset
        # are contained in `TrainConfiger`.
        # The engine will be started when `.run()` is called, and when `.terminate()` is called,
        # it will be completely terminated after the current iteration is finished.
        self.train_engine = conf.train_engine
        self.eval_engine = conf.eval_engine

    def assign_current_model(self, model_weights: Dict[str, np.ndarray]):
        """
        This function is used to load provided weights for the network.
        Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        More info of HE:
        https://github.com/NVIDIA/clara-train-examples/blob/master/PyTorch/NoteBooks/FL/Homomorphic_Encryption.ipynb

        """
        net = self.train_engine.network

        local_var_dict = net.state_dict()
        model_keys = model_weights.keys()
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = model_weights[var_name]
                try:
                    local_var_dict[var_name] = torch.as_tensor(
                        np.reshape(weights, local_var_dict[var_name].shape)
                    )
                except Exception as e:
                    raise ValueError(
                        "Convert weight from {} failed with error: {}".format(
                            var_name, str(e)
                        )
                    )

        net.load_state_dict(local_var_dict)

    def extract_model(self) -> Dict[str, np.ndarray]:
        """
        This function is used to extract weights of the network.
        The extracted weights will be converted into a numpy array based dict.
        """
        net = self.train_engine.network
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

    def generate_shareable(self):
        """
        This function is used to generate a DXO instance.
        The instance can contain not only model weights, but also
        some additional information that clients want to share.
        """
        # update meta, NUM_STEPS_CURRENT_ROUND is needed for aggregation.
        if self.achieved_meta is None:
            meta = {MetaKey.NUM_STEPS_CURRENT_ROUND: self.current_iters}
        else:
            meta = self.achieved_meta
            meta[MetaKey.NUM_STEPS_CURRENT_ROUND] = self.current_iters
        return DXO(
            data_kind=DataKind.WEIGHTS,
            data=self.extract_model(),
            meta=meta,
        ).to_shareable()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """
        This function is an extended function from the super class.
        It is used to handle two events:

        1) `START_RUN`. At the start point of a FL experiment,
        necessary components should be initialized.
        2) `ABORT_TASK`, when this event is fired, the running engines
        should be terminated (this example uses MONAI engines to do train
        and validation, and the engines can be terminated from another thread.
        If the solution does not provide any way to interrupt/end the execution,
        handle this event is not feasible).


        Args:
            event_type: the type of event that will be fired. In MONAITrainer,
                only `START_RUN` and `END_RUN` need to be handled.
            fl_ctx: an `FLContext` object.

        """
        if event_type == EventType.START_RUN:
            self._initialize_trainer(fl_ctx)
        elif event_type == EventType.ABORT_TASK:
            # This event is fired to abort the current execution task. We are using the ignite engine to run the task.
            # Unfortunately the ignite engine does not support the abort of task right now. We have to wait until
            # the current task finishes.
            pass
        elif event_type == EventType.END_RUN:
            self.eval_engine.terminate()
            self.train_engine.terminate()

    def _abort_execution(self) -> Shareable:
        shareable = Shareable()
        shareable.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
        return shareable

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """
        This function is an extended function from the super class.
        As a supervised learning based trainer, the execute function will run
        evaluate and train engines based on model weights from `shareable`.
        After fininshing training, a new `Shareable` object will be submitted
        to server for aggregation.

        Args:
            task_name: decide which task will be executed.
            shareable: the `Shareable` object acheived from server.
            fl_ctx: the `FLContext` object achieved from server.
            abort_signal: if triggered, the training will be aborted. In order to interrupt the training/validation
                state, a separate is used to check the signal information every few seconds. The implementation is
                shown in the `handle_event` function.
        Returns:
            a new `Shareable` object to be submitted to server for aggregation.
        """
        if task_name == self._train_task_name:
            # convert shareable into DXO instance
            dxo = from_shareable(shareable)
            # check if dxo is valid.
            if not isinstance(dxo, DXO):
                self.log_exception(
                    fl_ctx, f"dxo excepted type DXO. Got {type(dxo)} instead."
                )
                shareable.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
                return shareable

            # ensure data kind is weights.
            if not dxo.data_kind == DataKind.WEIGHTS:
                self.log_exception(
                    fl_ctx,
                    f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.",
                )
                shareable.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
                return shareable

            # load weights from dxo
            self.assign_current_model(dxo.data)
            # collect meta from dxo
            self.achieved_meta = dxo.meta

            # set engine state max epochs.
            self.train_engine.state.max_epochs = (
                self.train_engine.state.epoch + self.aggregation_epochs
            )
            # get current iteration when a round starts
            iter_of_start_time = self.train_engine.state.iteration

            # execute validation at the beginning of every round
            self.eval_engine.run(self.train_engine.state.epoch + 1)

            # check abort signal after validation
            if abort_signal.triggered:
                return self._abort_execution()

            self.train_engine.run()

            # check abort signal after train
            if abort_signal.triggered:
                return self._abort_execution()

            # calculate current iteration and epoch data after training.
            self.current_iters = self.train_engine.state.iteration - iter_of_start_time
            # create a new `Shareable` object
            return self.generate_shareable()
        else:
            # If unknown task name, set ReturnCode accordingly.
            shareable = Shareable()
            shareable.set_return_code(ReturnCode.TASK_UNKNOWN)
            return shareable
