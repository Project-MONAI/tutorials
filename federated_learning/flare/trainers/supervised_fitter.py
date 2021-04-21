import logging
import math

import torch.distributed as dist
from ignite.engine import Engine, Events


class StepAggregateHandler:
    """
    This class implements an event handler for step based aggregation.
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


class SupervisedFitter:
    """
    This class implements a supervised learning based fitter.
    """

    def __init__(
        self,
        trainer: Engine,
        evaluator: Engine,
        multi_gpu: bool,
        aggregation_epochs: int,
        aggregation_steps: int,
    ):
        if not isinstance(trainer, Engine):
            raise ValueError("trainer must be instance of PyTorch ignite Engine.")

        if aggregation_steps > 0:
            StepAggregateHandler(interval=aggregation_steps).attach(trainer)

        self.trainer = trainer
        self.evaluator = evaluator
        self.device = trainer.state.device
        self.multi_gpu = multi_gpu
        self.aggregation_epochs = aggregation_epochs
        self.aggregation_steps = aggregation_steps
        self.net = trainer.network

        self.train_ctx = TrainContext()
        self.train_ctx.my_rank = self.trainer.state.rank
        self.train_ctx.initial_learning_rate = self.get_lr_values(
            self.trainer.optimizer
        )
        self.train_ctx.num_gpus = dist.get_world_size() if multi_gpu else 1
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_lr_values(self, optimizer):
        return [group["lr"] for group in optimizer.state_dict()["param_groups"]]

    def get_train_context(self):
        self.train_ctx.total_steps = self.trainer.state.iteration
        self.train_ctx.total_executed_epochs = self.trainer.state.epoch
        self.train_ctx.current_learning_rate = self.get_lr_values(
            self.trainer.optimizer
        )
        self.train_ctx.fl_init_validation_metric = self.evaluator.state.best_metric

        return self.train_ctx

    def init_trainer(self):
        """
        This function is used to initialize a trainer.
        """
        if self.aggregation_steps > 0:
            next_aggr_step = self.trainer.state.iteration + self.aggregation_steps
            self.trainer.state.max_epochs = math.ceil(
                next_aggr_step / self.trainer.state.epoch_length
            )
            previous_step = (
                self.trainer.state.iteration % self.trainer.state.epoch_length
            )
            if self.trainer.state.iteration > 0 and previous_step != 0:
                # init to continue from previous epoch
                self.trainer.state.epoch -= 1
                if hasattr(self.trainer.state, "dataloader_iter"):
                    # init to continue from previous iteration
                    self.trainer._init_iter.append(previous_step)
                    self.trainer._dataloader_iter = self.trainer.state.dataloader_iter
        else:
            self.trainer.state.max_epochs = (
                self.trainer.state.epoch + self.aggregation_epochs
            )

    def fit(self):
        self.init_trainer()
        # get current epoch and iteration when a round starts
        self.train_ctx.epoch_of_start_time = self.trainer.state.epoch
        self.train_ctx.iter_of_start_time = self.trainer.state.iteration
        # execute validation at the beginning of every round
        self.evaluator.run(self.trainer.state.epoch + 1)

        starting_steps = self.trainer.state.iteration
        starting_epochs = self.trainer.state.epoch

        self.trainer.run()

        self.train_ctx.current_steps = self.trainer.state.iteration - starting_steps
        self.train_ctx.current_executed_epochs = (
            self.trainer.state.epoch - starting_epochs
        )
        return True

    def close(self):
        try:
            self.trainer.terminate()
            self.evaluator.terminate()
        except BaseException as e:
            self.logger.info(f"exception in closing fitter {e}")


class TrainContext:
    def __init__(self):
        self.training_impossible = False
        self.my_rank = 0
        self.initial_learning_rate = 0
        self.current_learning_rate = 0
        self.num_gpus = 1
        self.total_steps = 0
        self.total_executed_epochs = 0
        self.fl_init_validation_metric = 0
        self.epoch_of_start_time = 0
        self.iter_of_start_time = 0
