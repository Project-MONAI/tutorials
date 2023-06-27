# Monitoring and measuring GPU metrics with DCGM

## Introduction

[NVIDIA Data Center GPU Manager (DCGM)](https://developer.nvidia.com/dcgm) is a suite of tools for managing and monitoring NVIDIA datacenter GPUs in cluster environments. It includes active health monitoring, comprehensive diagnostics, system alerts and governance policies including power and clock management. It can be used standalone by infrastructure teams and easily integrates into cluster management tools, resource scheduling and monitoring products from NVIDIA partners. In this tutorial, we will provide basic examples to log the metrics.

## Installation

1. Follow [this guide](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/getting-started.html#supported-linux-distributions) and install datacenter-gpu-manager on a local or remote machine. The users can vserify the installation by `dcgmi discovery -l`.

2. Pull and run the `dcgm-exporter` container in this [link](https://github.com/NVIDIA/dcgm-exporter)) on the same machine to allow `curl`. For example:
```
DCGM_EXPORTER_VERSION=3.1.6-3.1.3 &&
docker run -itd --rm \
--gpus all \
--net host \
--cap-add SYS_ADMIN \
nvcr.io/nvidia/k8s/dcgm-exporter:${DCGM_EXPORTER_VERSION}-ubuntu20.04 \
-r localhost:5555 -f /etc/dcgm-exporter/dcp-metrics-included.csv -a ":<port>"
```
localhost:5555 points to the nv-host-engine in localhost. `<port>` has a default value of 9400 but the users can specify a number based on their environments. `/etc/dcgm-exporter/dcp-metrics-included.csv` is the list of metrics. It provides more information than the default one about the GPU usage.


## Quick start

After the docker is up for about 2-3 minutes, the user can use `curl` to get the GPU information on the "host-ip" machine `curl <host-ip>:<port>/metrics`

To use it in a container, the user can create a `log.sh` to start logging infinitely:
```
#!/bin/bash

set -e

file="output.log"
url="<host-ip>:<port>/metrics"

while true; do
  timestamp=$(date +%Y-%m-%d_%H:%M:%S)
  message=$(curl -s $url)
  echo -e "$timestamp:\n$message\n" >> $file
  sleep 30

done
```

```
DATETIME=$(date +"%Y%m%d-%H%M%S")
nohup ./log.sh >/dev/null 2>&1 & echo $! > log_process_${DATETIME}.pid
python ...
kill $(cat log_process_${DATETIME}.pid) && rm log_process_${DATETIME}.pid
```

The GPU utilization, as well as other metrics such as DRAM use, PCI Tx/Rx, can be found in the `output.log`. Depending on the GPU models, the following metrics may be recorded:

- DCGM_FI_PROF_GR_ENGINE_ACTIVE (GPU utilization if the arch supports)
- DCGM_FI_PROF_PIPE_TENSOR_ACTIVE (Tensor Core utilization)
- DCGM_FI_PROF_PCIE_TX_BYTES (PCI Transmit)
- DCGM_FI_PROF_PCIE_RX_BYTES (PCI Receiving)
- DCGM_FI_DEV_FB_USED (GPU memory usage)

## GPU Utilization

The overview of [NVIDIA DCGM Documentation](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/index.html) provides a detailed description on how to read the GPU utlization in the [metrics section](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html?#metrics).

Typically, the “GPU Utilization” from `nvidia-smi` or `NVML` is a rough metric that reflects how busy GPU cores are utilized. It is defined by “Percent of time over the past sample period during which one or more kernels was executing on the GPU”. In extreme cases, the metric is 100% even there’s only one thread launched to run kernel on GPU during past sample period.
