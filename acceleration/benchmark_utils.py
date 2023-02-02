import ast
import glob
import os
import tempfile
import time

import torch
import torch.profiler
import torch_tensorrt
from typing import Union, Sequence, Tuple

import matplotlib.pyplot as plt
import monai
from monai.apps import download_and_extract
from torch.jit._script import ScriptModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_current_device():
    return device


def get_precision_str(convert_precision):
    return "fp16" if convert_precision == torch.half else "fp32"


def _get_var_names(expr: str):
    """
    Parse the expression and discover what variables are present in it based on ast module.

    Args:
        expr: source expression to parse.

    """
    tree = ast.parse(expr)
    return [m.id for m in ast.walk(tree) if isinstance(m, ast.Name)]


def _get_fake_spatial_shape(shape: Sequence[Union[str, int]], p: int = 1, n: int = 1, any: int = 1) -> Tuple:
    """
    Get spatial shape for fake data according to the specified shape pattern.
    It supports `int` number and `string` with formats like: "32", "32 * n", "32 ** p", "32 ** p *n".

    Args:
        shape: specified pattern for the spatial shape.
        p: power factor to generate fake data shape if dim of expected shape is "x**p", default to 1.
        p: multiply factor to generate fake data shape if dim of expected shape is "x*n", default to 1.
        any: specified size to generate fake data shape if dim of expected shape is "*", default to 1.

    """
    ret = []
    for i in shape:
        if isinstance(i, int):
            ret.append(i)
        elif isinstance(i, str):
            if i == "*":
                ret.append(any)
            else:
                for c in _get_var_names(i):
                    if c not in ["p", "n"]:
                        raise ValueError(f"only support variables 'p' and 'n' so far, but got: {c}.")
                ret.append(eval(i, {"p": p, "n": n}))
        else:
            raise ValueError(f"spatial shape items must be int or string, but got: {type(i)} {i}.")
    return tuple(ret)


def get_timing_image_name(bundle_path, is_torch_module, precision_str):
    image_path = os.path.join(bundle_path, "eval")
    os.makedirs(image_path, exist_ok=True)
    if is_torch_module:
        image_name = os.path.join(image_path, "time_list_torch.png")
    else:
        image_name = os.path.join(image_path, f"time_list_{precision_str}.png")
    return image_name


def profile_random(
    bundle_path, model, input_shape, input_precision, wait_iter=2, warmup_iter=300, active_iter=1000, repeat_time=2
):
    total_iter = (wait_iter + warmup_iter + active_iter) * repeat_time
    precision_str = get_precision_str(input_precision)
    bundle_name = os.path.basename(bundle_path)
    bundle_log_dir = os.path.join(bundle_path, "log")
    is_torch_module = not isinstance(model, ScriptModule)
    tensorboard_log_dir = bundle_name + "torch" if is_torch_module else "trt" + precision_str
    tensorboard_log_dir = os.path.join(bundle_log_dir, tensorboard_log_dir)
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=wait_iter, warmup=warmup_iter, active=active_iter, repeat=repeat_time),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tensorboard_log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(total_iter):
            random_input = torch.rand(input_shape, dtype=torch.float32, device=device)
            pred = model(random_input)
            prof.step()


def inference_random_python_timer(bundle_path, model, input_shape, input_precision, warmup_iter=20, active_iter=50):
    precision_str = get_precision_str(input_precision)

    is_torch_module = not isinstance(model, ScriptModule)
    image_name = get_timing_image_name(bundle_path, is_torch_module, precision_str)

    for _ in range(warmup_iter):
        random_input = torch.rand(input_shape, dtype=torch.float32, device=device)
        pred = model(random_input)

    timeaccumulate = []
    for _ in range(active_iter):
        torch.cuda.synchronize()
        starter = time.time()
        pred = model(random_input)
        torch.cuda.synchronize()
        ender = time.time()
        time_cur = (ender - starter) * 1000
        timeaccumulate.append(time_cur)
    total_time = sum(timeaccumulate)
    average_time = total_time / (len(timeaccumulate) + 1e-12)
    print(f"Total time {total_time}ms. Average time {average_time}ms.")
    plt.plot(timeaccumulate)
    plt.savefig(image_name)
    return average_time


def inference_random_torch_timer(bundle_path, model, input_shape, input_precision, warmup_iter=500, active_iter=500):
    precision_str = get_precision_str(input_precision)

    is_torch_module = not isinstance(model, ScriptModule)
    image_name = get_timing_image_name(bundle_path, is_torch_module, precision_str)

    for _ in range(warmup_iter):
        random_input = torch.rand(input_shape, dtype=torch.float32, device=device)
        pred = model(random_input)

    timeaccumulate = []
    for _ in range(active_iter):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter.record()
        pred = model(random_input)
        ender.record()
        torch.cuda.synchronize()
        time_cur = starter.elapsed_time(ender)
        timeaccumulate.append(time_cur)
    total_time = sum(timeaccumulate)
    average_time = total_time / (len(timeaccumulate) + 1e-12)
    print(f"Total time {total_time}ms. Average time {average_time}ms.")
    plt.plot(timeaccumulate)
    plt.savefig(image_name)
    return average_time


def download_dataset(resource, md5, compressed_file_name, dst_dirname):
    # Setup data directory
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(f"root dir is: {root_dir}")

    # Download dataset
    # resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
    # md5 = "410d4a301da4e5b2f6f86ec3ddba524e"

    compressed_file = os.path.join(root_dir, compressed_file_name)
    data_root = os.path.join(root_dir, dst_dirname)
    if not os.path.exists(data_root):
        download_and_extract(resource, compressed_file, root_dir, md5)


def get_bundle_parser(bundle_path):
    meta_config = glob.glob(os.path.join(bundle_path, "*", "metadata.*"))[0]
    inference_config = glob.glob(os.path.join(bundle_path, "*", "inference.*"))[0]
    parser = monai.bundle.ConfigParser()
    parser.read_meta(meta_config)
    parser.read_config(inference_config)
    return parser


def get_pretrained_weight_path(bundle_path):
    return glob.glob(os.path.join(bundle_path, "*", "model.pt"))[0]


def find_network_id(parser):
    if "gnetwork" in parser:
        net_id = "gnetwork"
    elif "net" in parser:
        net_id = "net"
    elif "network_def" in parser:
        net_id = "network_def"
    else:
        net_id = "network"
    return net_id


def get_bundle_network(bundle_path, pretrained=True):
    model_weight = get_pretrained_weight_path(bundle_path)
    parser = get_bundle_parser(bundle_path)

    # If using other bundles, the net_id may be change. Please modify this variable accordingly.
    net_id = find_network_id(parser)
    model = parser.get_parsed_content(net_id)
    if pretrained and model_weight:
        model.load_state_dict(torch.load(model_weight))
    return model


def get_bundle_input_shape(bundle_path):
    parser = get_bundle_parser(bundle_path)

    if "_meta_#network_data_format#inputs#latent" in parser:
        input_channels = parser["_meta_#network_data_format#inputs#latent#num_channels"]
        input_spatial_shape = parser["_meta_#network_data_format#inputs#latent#spatial_shape"]
    else:
        input_channels = parser["_meta_#network_data_format#inputs#image#num_channels"]
        input_spatial_shape = parser["_meta_#network_data_format#inputs#image#spatial_shape"]
    spatial_shape = _get_fake_spatial_shape(input_spatial_shape)
    if not input_channels:
        input_shape = (1, *spatial_shape)
    else:
        input_shape = (1, input_channels, *spatial_shape)
    return input_shape


def convert_model_to_torchscript(model, input_shape, convert_precision):
    model.to(device)
    model.eval()
    with torch.no_grad():
        jit_model = torch.jit.script(model)

    inputs = [
        torch_tensorrt.Input(
            min_shape=input_shape,
            opt_shape=input_shape,
            max_shape=input_shape,
            dtype=torch.float,
        )
    ]
    enabled_precision = {convert_precision}
    with torch_tensorrt.logging.graphs():
        trt_ts_model = torch_tensorrt.compile(jit_model, inputs=inputs, enabled_precisions=enabled_precision)
    return trt_ts_model


def get_bundle_trt_model(bundle_path, model, input_shape, convert_precision, save_file=True):
    precision_string = get_precision_str(convert_precision)
    trt_model_file = os.path.join(bundle_path, "models", f"model_trt_{precision_string}.ts")
    if os.path.exists(trt_model_file):
        trt_model = torch.jit.load(trt_model_file)
    else:
        trt_model = convert_model_to_torchscript(model, input_shape, convert_precision)
        model_path = os.path.dirname(trt_model_file)
        os.makedirs(model_path, exist_ok=True)
        if save_file:
            torch.jit.save(trt_model, trt_model_file)
    return trt_model


class BundleBenchmark:
    def __init__(self):
        self.model = None
        self.dataloader = None
        self.timer = None
