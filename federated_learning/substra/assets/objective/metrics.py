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

import substratools as tools
import torch

from monai.metrics import DiceMetric


class MonaiMetrics(tools.Metrics):
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    def score(self, y_true, y_pred):
        metric_sum = 0.0
        metric_count = 0
        with torch.no_grad():
            for (val_true, val_pred) in zip(y_true, y_pred):
                val_true, _ = val_true
                val_pred, _ = val_pred
                value = self.dice_metric(
                    y_pred=val_pred,
                    y=val_true,
                )
                metric_count += len(value)
                metric_sum += value.item() * len(value)
        metric = metric_sum / metric_count
        return metric


if __name__ == "__main__":
    tools.metrics.execute(MonaiMetrics())
