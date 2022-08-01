# Run using Docker Container

## Build Docker Images

### Build Director Image

```
$ cd director\docker\cpu_intel
$ docker build -t openfl-monai-director .
```

```
$ docker images
```
```
REPOSITORY                      TAG            IMAGE ID       CREATED         SIZE
openfl-monai-director           latest         4c0b56c25b01   3 seconds ago   3.92GB
intel/intel-optimized-pytorch   1.11.0-conda   95e46843f5d3   2 months ago    2.82GB
```

### Build Envoy Image

```
$ cd envoy\docker\cpu_intel
$ docker build -t openfl-monai-envoy .
```

```
$ docker images
```
```
REPOSITORY                      TAG            IMAGE ID       CREATED         SIZE
openfl-monai-envoy              latest         b84b3e1e8cec   9 seconds ago   6.73GB
openfl-monai-director           latest         4c0b56c25b01   4 minutes ago   3.92GB
intel/intel-optimized-pytorch   1.11.0-conda   95e46843f5d3   2 months ago    2.82GB
```

### Build Workspace Image

```
$ cd workspace\docker\cpu_intel
$ docker build -t openfl-monai-workspace .
```

```
$ docker images
```
```
REPOSITORY                      TAG            IMAGE ID       CREATED         SIZE
openfl-monai-workspace          latest         482db72b2394   4 seconds ago   7.63GB
openfl-monai-envoy              latest         b84b3e1e8cec   4 minutes ago   6.73GB
openfl-monai-director           latest         4c0b56c25b01   8 minutes ago   3.92GB
intel/intel-optimized-pytorch   1.11.0-conda   95e46843f5d3   2 months ago    2.82GB
```

## Perform Training

### Run Director

Run the container.

```
$ docker run -it \
  --shm-size=8g \
  --network=host \
  --rm openfl-monai-director:latest /bin/bash
```

Inside the container, activate virtual environment.

```
$ cd /home/tami/Workspace/codes/tutorials/federated_learning/openfl/openfl_mednist_2d_registration/
$ source director_env/bin/activate
```

Start director.

```
$ cd director
$ fx director start --disable-tls --director-config-path director_config.yaml
```
```
[15:36:56] INFO     🧿 Starting the Director Service.                                          director.py:49
           INFO     Sample shape: ['64', '64', '1'], target shape: ['64', '64', '1']           director.py:58
           INFO     Starting server on localhost:50051                                 director_server.py:103
```

### Run Envoy

Run the container.

```
$ docker run -it \
  --shm-size=8g \
  --network=host \
  --rm openfl-monai-envoy:latest /bin/bash
```

Inside the container, activate virtual environment.

```
$ cd /home/tami/Workspace/codes/tutorials/federated_learning/openfl/openfl_mednist_2d_registration/
$ source envoy_env/bin/activate
```

Start the first envoy.

```
$ cd envoy
$ fx envoy start --shard-name env_one --disable-tls --envoy-config-path envoy_config_cpu_one.yaml --director-host localhost --director-port 50051
```

Run another container and start the second envoy.
```
$ cd envoy
$ fx envoy start --shard-name env_two --disable-tls --envoy-config-path envoy_config_cpu_two.yaml --director-host localhost --director-port 50051
```
```
[13:48:42] INFO     🧿 Starting the Envoy.                                                                                                            envoy.py:53
Downloading...
From: https://drive.google.com/uc?id=1QsnnkvZyJPcbRoV_ArW8SnE1OTuoVbKE
To: /tmp/tmpd60wcnn8/MedNIST.tar.gz
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 61.8M/61.8M [00:04<00:00, 13.8MB/s]
2022-07-22 13:48:48,735 - INFO - Downloaded: MedNIST.tar.gz
2022-07-22 13:48:48,816 - INFO - Verified 'MedNIST.tar.gz', md5: 0bc7306e7427e00ad1c5526a6677552d.
2022-07-22 13:48:48,817 - INFO - Writing into directory: ..
Loading dataset: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 47164/47164 [00:00<00:00, 199042.27it/s]
2022-07-22 13:48:53,669 - INFO - Verified 'MedNIST.tar.gz', md5: 0bc7306e7427e00ad1c5526a6677552d.
2022-07-22 13:48:53,669 - INFO - File exists: MedNIST.tar.gz, skipped downloading.
2022-07-22 13:48:53,669 - INFO - Non-empty folder exists in MedNIST, skipped extracting.
Loading dataset: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 5895/5895 [00:00<00:00, 281216.77it/s]
[13:48:53] INFO     Send report AcknowledgeShard                                                                                           director_client.py:58
           INFO     Shard accepted                                                                                                                  envoy.py:154
           INFO     The health check sender is started.                                                                                              envoy.py:92
           INFO     Send WaitExperiment request                                                                                            director_client.py:77
           INFO     WaitExperiment response has received                                                                                   director_client.py:79
```

### Run Workspace

Run the container.

```
$ docker run -it \
  --shm-size=8g \
  --network=host \
  --rm openfl-monai-workspace:latest /bin/bash
```

Inside the container, activate virtual environment.

```
$ cd /home/tami/Workspace/codes/tutorials/federated_learning/openfl/openfl_mednist_2d_registration/
$ source workspace_env/bin/activate
```

Start the notebook.

```
$ cd workspace
$ jupyter notebook --ip=0.0.0.0
```
```
[I 17:16:59.967 NotebookApp] Writing notebook server cookie secret to /home/tami/.local/share/jupyter/runtime/notebook_cookie_secret
[I 17:17:00.166 NotebookApp] Serving notebooks from local directory: /home/tami/Workspace/codes/tutorials/federated_learning/openfl/openfl_mednist_2d_registration/workspace
[I 17:17:00.166 NotebookApp] Jupyter Notebook 6.4.12 is running at:
[I 17:17:00.166 NotebookApp] http://2d967a81b3c5:8888/?token=c62...52d
[I 17:17:00.166 NotebookApp]  or http://127.0.0.1:8888/?token=c62...52d
[I 17:17:00.166 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[W 17:17:00.168 NotebookApp] No web browser found: could not locate runnable browser.
[C 17:17:00.168 NotebookApp]

    To access the notebook, open this file in a browser:
        file:///home/tami/.local/share/jupyter/runtime/nbserver-9-open.html
    Or copy and paste one of these URLs:
        http://2d967a81b3c5:8888/?token=c62...52d
     or http://127.0.0.1:8888/?token=c62...52d
```

Access the notebook server from local machine and execute
`Monai_MedNIST.ipynb` notebook.
