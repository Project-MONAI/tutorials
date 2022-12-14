# Overview
This tutorial shows how to extend the features of workflow in the model-zoo bundles based on `event-handler` mechanism.
Here we try to add the execution time computation logic in the spleen segmentation bundle.

## Event-handler mechanism
The bundles in the `model-zoo` are constructed by MONAI workflow, which can enable quick start of training and evaluation experiments.
The MONAI workflow is compatible with pytorch-ignite `Engine` and `Event-Handler` mechanism: https://pytorch-ignite.ai/.

So we can easily extend new features to the workflow by defining a new independent event handler and attaching to the workflow engine.

Here is all the supported `Event` in MONAI:
| Module | Event name | Description |
| --- | --- | --- |
| ignite.engine.Events | STARTED | triggered when engine's run is started |
| ignite.engine.Events | EPOCH_STARTED | triggered when the epoch is started |
| ignite.engine.Events | GET_BATCH_STARTED | triggered before next batch is fetched |
| ignite.engine.Events | GET_BATCH_COMPLETED | triggered after the batch is fetched |
| ignite.engine.Events | ITERATION_STARTED | triggered when an iteration is started |
| monai.engines.IterationEvents | FORWARD_COMPLETED | triggered when `network(image, label)` is completed |
| monai.engines.IterationEvents | LOSS_COMPLETED | triggered when `loss(pred, label)` is completed |
| monai.engines.IterationEvents | BACKWARD_COMPLETED | triggered when `loss.backward()` is completed |
| monai.engines.IterationEvents | MODEL_COMPLETED | triggered when all the model related operations completed |
| monai.engines.IterationEvents | INNER_ITERATION_STARTED | triggered when the iteration has an inner loop and the loop is started |
| monai.engines.IterationEvents | INNER_ITERATION_COMPLETED | triggered when the iteration has an inner loop and the loop is completed |
| ignite.engine.Events | ITERATION_COMPLETED | triggered when the iteration is ended |
| ignite.engine.Events | DATALOADER_STOP_ITERATION | triggered when dataloader has no more data to provide |
| ignite.engine.Events | EXCEPTION_RAISED | triggered when an exception is encountered |
| ignite.engine.Events | TERMINATE_SINGLE_EPOCH | triggered when the run is about to end the current epoch |
| ignite.engine.Events | TERMINATE | triggered when the run is about to end completely |
| ignite.engine.Events | INTERRUPT | triggered when the run is interrupted |
| ignite.engine.Events | EPOCH_COMPLETED | triggered when the epoch is ended |
| ignite.engine.Events | COMPLETED | triggered when engine's run is completed |

For more information about the `Event` of pytorch-ignite, please refer to:
https://pytorch.org/ignite/generated/ignite.engine.events.Events.html

Users can also register their own customized `Event` to the workflow engine.

A typical handler must contain the `attach()` function and several callback functions to handle the attached events.
For example, here we define a dummy handler to do some logic when iteration started and completed for every 5 iterations:
```py
from ignite.engine import Engine, Events


class DummyHandler:
    def attach(self, engine: Engine) -> None:
        engine.add_event_handler(Events.ITERATION_STARTED(every=5), self.iteration_started)
        engine.add_event_handler(Events.ITERATION_COMPLETED(every=5), self.iteration_completed)

    def iteration_started(self, engine: Engine) -> None:
        pass

    def iteration_completed(self, engine: Engine) -> None:
        pass
```

## Download example MONAI bundle from model-zoo
```
python -m monai.bundle download --name spleen_ct_segmentation --version "0.1.1" --bundle_dir "./"
```

## Extend the workflow to print the execution time for every iteration, every epoch and total time
Here we define a new handler in `spleen_ct_segmentation/scripts/timer.py` to compute and print the time consumption details:
```py
from time import time
from ignite.engine import Engine, Events


class TimerHandler:
    def __init__(self) -> None:
        self.start_time = 0
        self.epoch_start_time = 0
        self.iteration_start_time = 0

    def attach(self, engine: Engine) -> None:
        engine.add_event_handler(Events.STARTED, self.started)
        engine.add_event_handler(Events.EPOCH_STARTED, self.epoch_started)
        engine.add_event_handler(Events.ITERATION_STARTED, self.iteration_started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.epoch_completed)
        engine.add_event_handler(Events.COMPLETED, self.completed)

    def started(self, engine: Engine) -> None:
        self.start_time = time()

    def epoch_started(self, engine: Engine) -> None:
        self.epoch_start_time = time()

    def iteration_started(self, engine: Engine) -> None:
        self.iteration_start_time = time()

    def iteration_completed(self, engine: Engine) -> None:
        print(f"iteration {engine.state.iteration} execution time: {time() - self.iteration_start_time}")

    def epoch_completed(self, engine: Engine) -> None:
        print(f"epoch {engine.state.epoch} execution time: {time() - self.epoch_start_time}")

    def completed(self, engine: Engine) -> None:
        print(f"total execution time: {time() - self.start_time}")
```
Then add the handler to the `"train": {"handlers: [...]"}` list of `train.json` config:
```json
{
    "_target_": "scripts.timer.TimerHandler"
}
```

## Command example
To run the workflow with this customized handler, `PYTHONPATH` should be revised to include the path to the customized scripts:
```
export PYTHONPATH=$PYTHONPATH:"<path to 'spleen_ct_segmentation/scripts'>"
```
And please make sure the folder `spleen_ct_segmentation/scripts` is a valid python module (it has a `__init__.py` file in the folder).

Execute training:

```
python -m monai.bundle run training --meta_file configs/metadata.json --config_file configs/train.json --logging_file configs/logging.conf
```
