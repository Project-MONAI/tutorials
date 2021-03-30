from typing import Any, Dict, Tuple

import torch
from ignite.engine import Engine
from monai.engines import SupervisedTrainer
from monai.engines.utils import CommonKeys as Keys
from monai.engines.utils import IterationEvents
from torch.nn.parallel import DistributedDataParallel


class DynUNetTrainer(SupervisedTrainer):
    """
    This class inherits from SupervisedTrainer in MONAI, and is used with DynUNet
    on Decathlon datasets.

    """

    def _iteration(self, engine: Engine, batchdata: Dict[str, Any]):
        """
        Callback function for the Supervised Training processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - PRED: prediction result of model.
            - LOSS: loss value computed by loss function.

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
            inputs, targets = batch
            args: Tuple = ()
            kwargs: Dict = {}
        else:
            inputs, targets, args, kwargs = batch
        # put iteration outputs into engine.state
        engine.state.output = output = {Keys.IMAGE: inputs, Keys.LABEL: targets}

        def _compute_pred_loss():
            preds = self.inferer(inputs, self.network, *args, **kwargs)
            if len(preds.size()) - len(targets.size()) == 1:
                # deep supervision mode, need to unbind feature maps first.
                preds = torch.unbind(preds, dim=1)
            output[Keys.PRED] = preds
            del preds
            engine.fire_event(IterationEvents.FORWARD_COMPLETED)
            output[Keys.LOSS] = sum(
                0.5 ** i * self.loss_function.forward(p, targets)
                for i, p in enumerate(output[Keys.PRED])
            )
            engine.fire_event(IterationEvents.LOSS_COMPLETED)

        self.network.train()
        self.optimizer.zero_grad()
        if self.amp and self.scaler is not None:
            with torch.cuda.amp.autocast():
                _compute_pred_loss()
            self.scaler.scale(output[Keys.LOSS]).backward()
            self.scaler.unscale_(self.optimizer)
            if isinstance(self.network, DistributedDataParallel):
                torch.nn.utils.clip_grad_norm_(self.network.module.parameters(), 12)
            else:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            _compute_pred_loss()
            output[Keys.LOSS].backward()
            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
            if isinstance(self.network, DistributedDataParallel):
                torch.nn.utils.clip_grad_norm_(self.network.module.parameters(), 12)
            else:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
            engine.fire_event(IterationEvents.OPTIMIZER_COMPLETED)

        return output
