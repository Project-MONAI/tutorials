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

from monai.apps.auto3dseg import AutoRunner

# the minimum required code is to create an AutoRunner() and call runner.run()
# the algos must be set to 'segresnet' (since currently it's the only algo with support of multi-resolution input images, such as CT and PET)
# here we also set ensemble=False (optional) to prevent inference on the testing set (since we do not use any testing sets, only the 5-fold cross validation)
# for you own inference (and ensemble) you can provide a list of testing files in "hecktor22_folds.json"
runner = AutoRunner(input='input.yaml', algos = 'segresnet', work_dir= './work_dir', ensemble=False)

## optionally, we can use just 1-fold (for a quick training of a single model, instead of training 5 folds)
# runner.set_num_fold(1)

## optionally, we can define the path to the dataset here, instead of the one in input.yaml
#runner.set_training_params({"dataroot" : '/data/hecktor22'})

runner.run()
