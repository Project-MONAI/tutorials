# Copyright 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import numpy as np
import torch
import torch.optim as optim
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import TorchVisionFCModel
from monai.transforms import (
    Compose,
    EnsureTyped,
    LoadImaged,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
    Transposed,
)
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants, ModelName, ValidateType
from sklearn.metrics import cohen_kappa_score
from torch.utils.tensorboard import SummaryWriter


def load_datalist(filename, data_list_key="train", base_dir=""):
    with open(filename, "r") as f:
        data = json.load(f)

    data_list = data[data_list_key]
    for d in data_list:
        d["image"] = os.path.join(base_dir, d["image"])

    return data_list


class MammoLearner(Learner):
    def __init__(
        self,
        dataset_root: str = None,
        datalist_prefix: str = None,
        aggregation_epochs: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
        lr: float = 1e-2,
        batch_size: int = 32,
        val_freq: int = 1,
        val_frac: float = 0.1,
        analytic_sender_id: str = "analytic_sender",
    ):
        """Simple CIFAR-10 Trainer.

        Args:
            dataset_root: directory with breast density mammography data.
            datalist_prefix: json file with data list
            aggregation_epochs: the number of training epochs for a round. Defaults to 1.
            train_task_name: name of the task to train the model.
            submit_model_task_name: name of the task to submit the best local model.
            lr: local learning rate. Float number. Defaults to 1e-2.
            val_freq: int. How often to validate during local training
            val_frac: float. Fraction of training set to reserve for validation/model selection
            analytic_sender_id: id of `AnalyticsSender` if configured as a client component. If configured, TensorBoard events will be fired. Defaults to "analytic_sender".
        Returns:
            a Shareable with the updated local model after running `execute()`
            or the best local model depending on the specified task.
        """
        super().__init__()
        # trainer init happens at the very beginning, only the basic info regarding the trainer is set here
        # the actual run has not started at this point
        self.dataset_root = dataset_root
        self.datalist_prefix = datalist_prefix
        self.aggregation_epochs = aggregation_epochs
        self.train_task_name = train_task_name
        self.lr = lr
        self.batch_size = batch_size
        self.val_freq = val_freq
        self.submit_model_task_name = submit_model_task_name
        self.best_metric = 0.0
        self.val_frac = val_frac
        self.analytic_sender_id = analytic_sender_id

        # Epoch counter
        self.epoch_of_start_time = 0
        self.epoch_global = 0

        if not isinstance(self.val_freq, int):
            raise ValueError(f"Expected `val_freq` but got type {type(self.val_freq)}")

        # The following objects will be build in `initialize()`
        self.app_root = None
        self.client_id = None
        self.local_model_file = None
        self.best_local_model_file = None
        self.writer = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.transform_train = None
        self.transform_valid = None
        self.transform_test = None
        self.train_dataset = None
        self.train_loader = None
        self.valid_dataset = None
        self.valid_loader = None
        self.test_dataset = None
        self.test_loader = None

    def initialize(self, parts: dict, fl_ctx: FLContext):
        # when the run starts, this is where the actual settings get initialized for trainer

        # Set the paths according to fl_ctx
        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} initialized at \n {self.app_root} \n with args: {fl_args}",
        )

        self.local_model_file = os.path.join(self.app_root, "local_model.pt")
        self.best_local_model_file = os.path.join(self.app_root, "best_local_model.pt")

        # Select local TensorBoard writer or event-based writer for streaming
        self.writer = parts.get(
            self.analytic_sender_id
        )  # user configured config_fed_client.json for streaming
        if not self.writer:  # use local TensorBoard writer only
            self.writer = SummaryWriter(self.app_root)

        # set the training-related parameters
        # can be replaced by a config-style block
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = TorchVisionFCModel(
            "resnet18", n_classes=4, use_conv=False, pretrained=False, pool=None
        )  # pretrained is used only on server
        self.model = self.model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(self.device)

        self.transform_train = Compose(
            [
                LoadImaged(keys=["image"]),
                Transposed(keys=["image"], indices=[2, 0, 1]),  # make channels-first
                RandRotated(
                    keys=["image"], range_x=np.pi / 12, prob=0.5, keep_size=True
                ),
                RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
                RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
                RandZoomd(
                    keys=["image"], min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True
                ),
                RandGaussianSmoothd(
                    keys=["image"],
                    sigma_x=(0.5, 1.15),
                    sigma_y=(0.5, 1.15),
                    sigma_z=(0.5, 1.15),
                    prob=0.15,
                ),
                RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.5),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
                RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
        self.transform_valid = Compose(
            [
                LoadImaged(keys=["image"]),
                Transposed(keys=["image"], indices=[2, 0, 1]),  # make channels-first
                EnsureTyped(keys=["image", "label"]),
            ]
        )
        self.transform_test = Compose(
            [
                LoadImaged(keys=["image"]),
                Transposed(keys=["image"], indices=[2, 0, 1]),  # make channels-first
                EnsureTyped(keys=["image"]),  # Testing set won't have labels
            ]
        )

        # Note, do not change this syntax. The data list filename is given by the system.
        datalist_file = self.datalist_prefix + self.client_id + ".json"
        if not os.path.isfile(datalist_file):
            self.log_critical(fl_ctx, f"{datalist_file} does not exist!")

        # Set dataset
        train_datalist = load_datalist(
            datalist_file,
            data_list_key="train",  # do not change this key name
            base_dir=self.dataset_root,
        )

        # Validation set can be created from training set.
        if self.val_frac > 0:
            np.random.seed(0)
            val_indices = np.random.randint(
                0, len(train_datalist), size=int(self.val_frac * len(train_datalist))
            )
            val_datalist = [train_datalist[i] for i in val_indices]
            train_indices = list(set(np.arange(len(train_datalist))) - set(val_indices))
            train_datalist = [
                train_datalist[i] for i in train_indices
            ]  # remove validation entries from training
            assert (len(np.intersect1d(val_indices, train_indices))) == 0
            self.log_info(
                fl_ctx,
                f"Reserved {len(val_indices)} entries for validation during training.",
            )
        elif self.val_frac >= 1.0:
            raise ValueError(
                f"`val_frac` was {self.val_frac}. Cannot use whole training set for validation, use 0 > `val_frac` < 1."
            )
        else:
            val_datalist = []

        test_datalist = load_datalist(
            datalist_file,
            data_list_key="test",  # do not change this key name
            base_dir=self.dataset_root,
        )

        num_workers = 1
        cache_rate = 1.0
        self.train_dataset = CacheDataset(
            data=train_datalist,
            transform=self.transform_train,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        self.log_info(fl_ctx, f"Training set: {len(train_datalist)} entries")

        if len(val_datalist) > 0:
            self.valid_dataset = CacheDataset(
                data=val_datalist,
                transform=self.transform_valid,
                cache_rate=cache_rate,
                num_workers=num_workers,
            )
            self.valid_loader = DataLoader(
                self.valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            self.log_info(fl_ctx, f"Validation set: {len(train_datalist)} entries")
        else:
            self.valid_dataset = None
            self.valid_loader = None
            self.log_info(fl_ctx, "Use no validation set")

        # evaluation on testing is required
        self.test_dataset = CacheDataset(
            data=test_datalist,
            transform=self.transform_test,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        self.log_info(fl_ctx, f"Testing set: {len(train_datalist)} entries")

        self.log_info(fl_ctx, f"Finished initializing {self.client_id}")

    def finalize(self, fl_ctx: FLContext):
        # collect threads, close files here
        pass

    def local_train(
        self, fl_ctx, train_loader, abort_signal: Signal, val_freq: int = 0
    ):
        for epoch in range(self.aggregation_epochs):
            if abort_signal.triggered:
                return
            self.model.train()
            epoch_len = len(train_loader)
            self.epoch_global = self.epoch_of_start_time + epoch
            self.log_info(
                fl_ctx,
                f"Local epoch {self.client_id}: {epoch + 1}/{self.aggregation_epochs} (lr={self.lr})",
            )
            avg_loss = 0.0
            for i, batch_data in enumerate(train_loader):
                if abort_signal.triggered:
                    return
                inputs, labels = (
                    batch_data["image"].to(self.device),
                    batch_data["label"].to(self.device),
                )
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()
                current_step = epoch_len * self.epoch_global + i
                avg_loss += loss.item()
            self.writer.add_scalar(
                "train_loss", avg_loss / len(train_loader), current_step
            )
            if val_freq > 0 and epoch % val_freq == 0:
                acc, kappa = self.local_valid(
                    self.valid_loader,
                    abort_signal,
                    tb_id="val_acc_local_model",
                    fl_ctx=fl_ctx,
                )
                if kappa > self.best_metric:
                    self.best_metric = kappa
                    self.save_model(is_best=True)

    def save_model(self, is_best=False):
        # save model
        model_weights = self.model.state_dict()
        save_dict = {"model_weights": model_weights, "epoch": self.epoch_global}
        if is_best:
            save_dict.update({"best_acc": self.best_metric})
            torch.save(save_dict, self.best_local_model_file)
        else:
            torch.save(save_dict, self.local_model_file)

    def train(
        self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal
    ) -> Shareable:
        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # get round information
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        self.log_info(
            fl_ctx, f"Current/Total Round: {current_round + 1}/{total_rounds}"
        )
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = global_weights[var_name]
                try:
                    # reshape global weights to compute difference later on
                    global_weights[var_name] = np.reshape(
                        weights, local_var_dict[var_name].shape
                    )
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(global_weights[var_name])
                except Exception as e:
                    raise ValueError(
                        "Convert weight from {} failed with error: {}".format(
                            var_name, str(e)
                        )
                    )
        self.model.load_state_dict(local_var_dict)

        # local steps
        epoch_len = len(self.train_loader)
        self.log_info(fl_ctx, f"Local steps per epoch: {epoch_len}")

        # local train
        self.local_train(
            fl_ctx=fl_ctx,
            train_loader=self.train_loader,
            abort_signal=abort_signal,
            val_freq=self.val_freq,
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.epoch_of_start_time += self.aggregation_epochs

        # perform valid after local train
        acc, kappa = self.local_valid(
            self.valid_loader, abort_signal, tb_id="val_local_model", fl_ctx=fl_ctx
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.log_info(fl_ctx, f"val_acc_local_model: {acc:.4f}")

        # save model
        self.save_model(is_best=False)
        if kappa > self.best_metric:
            self.best_metric = kappa
            self.save_model(is_best=True)

        # compute delta model, global model has the primary key set
        local_weights = self.model.state_dict()
        model_diff = {}
        for name in global_weights:
            if name not in local_weights:
                continue
            model_diff[name] = local_weights[name].cpu().numpy() - global_weights[name]
            if np.any(np.isnan(model_diff[name])):
                self.system_panic(f"{name} weights became NaN...", fl_ctx)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # build the shareable
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff)
        dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, epoch_len)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        return dxo.to_shareable()

    def get_model_for_validation(self, model_name: str, fl_ctx: FLContext) -> Shareable:
        # Retrieve the best local model saved during training.
        if model_name == ModelName.BEST_MODEL:
            model_data = None
            try:
                # load model to cpu as server might or might not have a GPU
                model_data = torch.load(self.best_local_model_file, map_location="cpu")
            except Exception as e:
                self.log_error(fl_ctx, f"Unable to load best model: {e}")

            # Create DXO and shareable from model data.
            if model_data:
                dxo = DXO(data_kind=DataKind.WEIGHTS, data=model_data["model_weights"])
                return dxo.to_shareable()
            else:
                # Set return code.
                self.log_error(
                    fl_ctx,
                    f"best local model not found at {self.best_local_model_file}.",
                )
                return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)
        else:
            raise ValueError(
                f"Unknown model_type: {model_name}"
            )  # Raised errors are caught in LearnerExecutor class.

    def local_valid(
        self,
        valid_loader,
        abort_signal: Signal,
        tb_id=None,
        return_probs_only=False,
        fl_ctx=None,
    ):
        if not valid_loader:
            return None
        self.model.eval()
        return_probs = []
        labels = []
        pred_labels = []
        with torch.no_grad():
            correct, total = 0, 0
            for i, batch_data in enumerate(valid_loader):
                if abort_signal.triggered:
                    return None
                inputs = batch_data["image"].to(self.device)
                outputs = torch.softmax(self.model(inputs), dim=1)
                probs = outputs.detach().cpu().numpy()
                # make json serializable
                for _img_file, _probs in zip(
                    batch_data["image_meta_dict"]["filename_or_obj"], probs
                ):
                    return_probs.append(
                        {
                            "image": os.path.basename(_img_file),
                            "probs": [float(p) for p in _probs],
                        }
                    )
                if not return_probs_only:
                    _, _pred_label = torch.max(outputs.data, 1)
                    _labels = batch_data["label"].to(self.device)
                    total += inputs.data.size()[0]
                    correct += (_pred_label == _labels.data).sum().item()
                    labels.extend(_labels.detach().cpu().numpy())
                    pred_labels.extend(_pred_label.detach().cpu().numpy())
            if return_probs_only:
                return return_probs  # create a list of image names and probs
            else:
                acc = correct / float(total)
                assert len(labels) == total
                assert len(pred_labels) == total
                kappa = cohen_kappa_score(labels, pred_labels, weights="linear")
                if tb_id:
                    self.writer.add_scalar(tb_id + "_acc", acc, self.epoch_global)
                    self.writer.add_scalar(tb_id + "_kappa", kappa, self.epoch_global)
                return acc, kappa

    def validate(
        self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal
    ) -> Shareable:
        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # get validation information
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")
        model_owner = shareable.get(ReservedHeaderKey.HEADERS).get(
            AppConstants.MODEL_OWNER
        )
        if model_owner:
            self.log_info(
                fl_ctx,
                f"Evaluating model from {model_owner} on {fl_ctx.get_identity_name()}",
            )
        else:
            model_owner = "global_model"  # evaluating global model during training

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        n_loaded = 0
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = torch.as_tensor(global_weights[var_name], device=self.device)
                try:
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(
                        torch.reshape(weights, local_var_dict[var_name].shape)
                    )
                    n_loaded += 1
                except Exception as e:
                    raise ValueError(
                        "Convert weight from {} failed with error: {}".format(
                            var_name, str(e)
                        )
                    )
        self.model.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError(
                f"No weights loaded for validation! Received weight dict is {global_weights}"
            )

        validate_type = shareable.get_header(AppConstants.VALIDATE_TYPE)
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            try:
                # perform valid before local train
                global_acc, global_kappa = self.local_valid(
                    self.valid_loader,
                    abort_signal,
                    tb_id="val_global_model",
                    fl_ctx=fl_ctx,
                )
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                self.log_info(
                    fl_ctx, f"val_acc_global_model ({model_owner}): {global_acc}"
                )

                return DXO(
                    data_kind=DataKind.METRICS,
                    data={MetaKey.INITIAL_METRICS: global_acc},
                    meta={},
                ).to_shareable()
            except Exception as e:
                raise ValueError(f"BEFORE_TRAIN_VALIDATE failed: {e}")
        elif validate_type == ValidateType.MODEL_VALIDATE:
            try:
                # perform valid
                train_acc, train_kappa = self.local_valid(
                    self.train_loader, abort_signal
                )
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                self.log_info(fl_ctx, f"training acc ({model_owner}): {train_acc}")

                val_acc, val_kappa = self.local_valid(self.valid_loader, abort_signal)

                # testing performance
                test_probs = self.local_valid(
                    self.test_loader, abort_signal, return_probs_only=True
                )
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                self.log_info(fl_ctx, f"validation acc ({model_owner}): {val_acc}")

                self.log_info(fl_ctx, "Evaluation finished. Returning shareable")

                val_results = {
                    "train_accuracy": train_acc,
                    "train_kappa": train_kappa,
                    "val_accuracy": val_acc,
                    "val_kappa": val_kappa,
                    "test_probs": test_probs,
                }

                metric_dxo = DXO(data_kind=DataKind.METRICS, data=val_results)
                return metric_dxo.to_shareable()
            except Exception as e:
                raise ValueError(f"MODEL_VALIDATE failed: {e}")
        else:
            return make_reply(ReturnCode.VALIDATE_TYPE_UNKNOWN)


# To test your Learner

class MockClientEngine:
    def __init__(self, run_num=0):
        self.fl_ctx_mgr = FLContextManager(
            engine=self,
            identity_name="site-1",
            run_num=run_num,
            public_stickers={},
            private_stickers={},
        )

    def new_context(self):
        return self.fl_ctx_mgr.new_context()

    def fire_event(self, event_type: str, fl_ctx: FLContext):
        pass


if __name__ == "__main__":
    inside_container = True
    if inside_container:
        debug_dataset_root = "/data/preprocessed"
        debug_datalist_prefix = "/data/dataset_blinded_phase1_"
    else:
        # assumes script is run in from repo root, e.g. using `python3 code/pt/learners/mammo_learner.py`
        debug_dataset_root = "./data/preprocessed"
        debug_datalist_prefix = "./data/dataset_blinded_phase1_"

    print("Testing MammoLearner...")
    learner = MammoLearner(
        dataset_root=debug_dataset_root,
        datalist_prefix=debug_datalist_prefix,
        aggregation_epochs=1,
        lr=1e-2,
    )
    engine = MockClientEngine()
    fl_ctx = engine.fl_ctx_mgr.new_context()
    fl_ctx.set_prop(FLContextKey.APP_ROOT, "/tmp/debug")

    print("test initialize...")
    learner.initialize(parts={}, fl_ctx=fl_ctx)

    print("test train...")
    learner.local_train(
        fl_ctx=fl_ctx,
        train_loader=learner.train_loader,
        abort_signal=Signal(),
        val_freq=1,
    )

    print("test valid...")
    acc, kappa = learner.local_valid(
        valid_loader=learner.valid_loader,
        abort_signal=Signal(),
        tb_id="val_debug",
        fl_ctx=fl_ctx,
    )
    print("debug acc", acc)
    print("debug kappa", kappa)

    print("test valid...")
    test_probs = learner.local_valid(
        valid_loader=learner.test_loader, abort_signal=Signal(), return_probs_only=True
    )
    print("test_probs", test_probs)

    print("finished testing.")

    # you can check the result for one epoch and validation on TensorBoard using
    # `tensorboard --logdir=./debug`
