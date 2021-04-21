import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from flare.apis.event_type import EventType
from flare.apis.fl_constant import FLConstants, ShareableKey, ShareableValue
from flare.apis.fl_context import FLContext
from flare.apis.shareable import Shareable
from flare.apis.trainer import Trainer
from flare.common.signal import Signal
from flare.utils.fed_utils import generate_failure

from .supervised_fitter import SupervisedFitter, TrainContext
from .train_configer import TrainConfiger


class MonaiModelReaderWriter:
    """
    This class is used to read/write model weights from/to FL context.
    The format of weights in Shareable should be numpy arrays. Therefore,
    torch tensor based weights need to be converted.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_model(self, fl_ctx: FLContext) -> Dict[str, np.ndarray]:
        """
        This function is used to extract model weights and converted them into
        a dict with numpy arrays and then return the dict.
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

    def apply_model(
        self, model_weights: Dict[str, np.ndarray], fl_ctx: FLContext
    ) -> List[Tuple]:
        """
        This function is used to load weights for network saved in FL context
        from provided dict. The function will return a list that contains the shapes
        for all updated layers.
        """
        net = fl_ctx.get_prop(FLConstants.MODEL_NETWORK)
        if fl_ctx.get_prop(FLConstants.MULTI_GPU):
            net = net.module

        local_var_dict = net.state_dict()
        changed_vars_shapes = []
        model_keys = model_weights.keys()
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = model_weights[var_name]
                try:
                    local_var_dict[var_name] = torch.as_tensor(weights)
                    changed_vars_shapes.append(weights.shape)
                except Exception as e:
                    raise ValueError(
                        "Convert weight from {} failed with error: {}".format(
                            var_name, str(e)
                        )
                    )

        net.load_state_dict(local_var_dict)
        return changed_vars_shapes


class MonaiModelManager:
    def __init__(self, model_reader_writer):
        self.model_reader_writer = model_reader_writer
        self.logger = logging.getLogger("ModelShareableManager")

    def assign_current_model(
        self, model_weights: Dict[str, np.ndarray], fl_ctx: FLContext
    ):
        """
        This function is used to load provided weights for the network from FL context.
        """
        changed_vars_shapes = self.model_reader_writer.apply_model(
            model_weights, fl_ctx
        )

        if len(changed_vars_shapes) == 0:
            return False
        num_vars = 0
        for changed_vars_shape in changed_vars_shapes:
            num_vars += np.prod(changed_vars_shape)
        self.logger.info("Setting global federated model data (%s elements)", num_vars)
        self.logger.info(
            "Round %s: local model updated", fl_ctx.get_prop(FLConstants.CURRENT_ROUND)
        )
        return True

    def generate_shareable(self, train_ctx: TrainContext, fl_ctx: FLContext):
        """
        This function is used to generate a shareable instance according to
        provided train context and FL context.
        """
        local_model_dict = self.model_reader_writer.extract_model(fl_ctx)

        meta_data = {}
        meta_data[FLConstants.NUM_TOTAL_STEPS] = train_ctx.total_steps
        meta_data[FLConstants.INITIAL_LEARNING_RATE] = train_ctx.initial_learning_rate
        meta_data[FLConstants.NUMBER_OF_GPUS] = train_ctx.num_gpus
        meta_data[FLConstants.LOCAL_EPOCHS] = train_ctx.total_executed_epochs
        meta_data[FLConstants.CURRENT_LEARNING_RATE] = train_ctx.current_learning_rate
        meta_data[FLConstants.INITIAL_METRICS] = train_ctx.fl_init_validation_metric
        meta_data[FLConstants.NUM_STEPS_CURRENT_ROUND] = train_ctx.current_steps
        meta_data[
            FLConstants.NUM_EPOCHS_CURRENT_ROUND
        ] = train_ctx.current_executed_epochs

        shareable = Shareable()
        shareable[ShareableKey.TYPE] = ShareableValue.TYPE_WEIGHT_DIFF
        shareable[ShareableKey.DATA_TYPE] = ShareableValue.DATA_TYPE_UNENCRYPTED
        shareable[ShareableKey.MODEL_WEIGHTS] = local_model_dict
        shareable[ShareableKey.META] = meta_data

        return shareable


class MonaiTrainer(Trainer):
    """
    This class implements a MONAI based trainer that can be used for Flare.

    Args:
        aggregation_epochs: the number of locally running epochs. This parameter
            only works when `aggregation_steps` is 0. Defaults to 1.
        aggregation_steps:  the number of locally running steps. If the value is
            larger than 0, the trainer will use step based aggregation rather
            than epoch based aggregation. Defaults to 0.
    """

    def __init__(self, aggregation_epochs: int = 1, aggregation_steps: int = 0):
        super().__init__()
        self.aggregation_epochs = aggregation_epochs
        self.aggregation_steps = aggregation_steps
        self.model_reader_writer = MonaiModelReaderWriter()
        self.model_manager = MonaiModelManager(self.model_reader_writer)
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_fitter(self):
        """
        This function is used to create a fitter.
        """
        self.fitter = SupervisedFitter(
            trainer=self.trainer,
            evaluator=self.evaluator,
            multi_gpu=self.multi_gpu,
            aggregation_epochs=self.aggregation_epochs,
            aggregation_steps=self.aggregation_steps,
        )

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """
        This function is used to perform the handler process based on the
        event_type.
        """
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type == EventType.END_RUN:
            self.finalize(fl_ctx)

    def initialize(self, fl_ctx: FLContext):
        """
        The initializing function.
        """
        config_root = fl_ctx.get_prop(FLConstants.TRAIN_ROOT)
        args = fl_ctx.get_prop(FLConstants.ARGS)

        conf = TrainConfiger(
            config_root=config_root,
            wf_config_file_name=args.train_config,
            local_rank=args.local_rank,
        )
        conf.configure()

        self.trainer = conf.trainer
        self.evaluator = conf.evaluator
        self.multi_gpu = conf.multi_gpu

        self.create_fitter()

        fl_ctx.set_prop(FLConstants.MY_RANK, self.trainer.state.rank)
        fl_ctx.set_prop(FLConstants.MODEL_NETWORK, self.fitter.net)
        fl_ctx.set_prop(FLConstants.MULTI_GPU, self.fitter.multi_gpu)
        fl_ctx.set_prop(FLConstants.DEVICE, self.fitter.device)

    def train(
        self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal
    ) -> Shareable:
        """
        The training function.
        """
        self.logger.info(f"MonaiTrainer abort signal: {abort_signal.triggered}")
        if abort_signal.triggered:
            self.finalize(fl_ctx)
            meta = shareable.get(ShareableKey.META)
            cookie = None
            if meta is not None:
                cookie = meta.get(FLConstants.META_COOKIE)
                self.logger.info(f"cookie {cookie}")
            shareable = generate_failure(
                fl_ctx, "abort signal triggered", cookie=cookie
            )
            return shareable
        if self.trainer.state.rank == 0:
            model_weights = shareable[ShareableKey.MODEL_WEIGHTS]
            _ = self.model_manager.assign_current_model(model_weights, fl_ctx)

        if self.fitter.multi_gpu:
            net = fl_ctx.get_prop(FLConstants.MODEL_NETWORK)
            for _, v in net.state_dict().items():
                dist.broadcast(v, src=0)

        self.fitter.fit()

        if self.trainer.state.rank == 0:
            shareable = self.model_manager.generate_shareable(
                self.fitter.get_train_context(),
                fl_ctx,
            )

        fl_ctx.set_prop(FLConstants.TRAIN_CONTEXT, self.fitter.get_train_context())
        return shareable

    def finalize(self, fl_ctx: FLContext):
        """
        The finalizing function.
        """
        if self.fitter:
            self.fitter.close()
