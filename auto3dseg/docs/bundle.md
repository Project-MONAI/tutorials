## Algorithm Bundle

We provide four default algorithms (DiNTS, 2D/3D SegResNet, SwinUNETR) as baseline algorithms of **Auto3DSeg**. Each algorithm includes advanced network architectures and well-tuned model training recipes. And the algorithms are formulated in the MONAI package format, with an independent folder structure, providing the necessary functions for model training, verification, and inference. The concept of the MONAI bundle can be found in the following [link](https://docs.monai.io/en/latest/mb_specification.html).

### Bundle Structure

The basic folder structure of algorithm bundles is shown as follows. The current bundle design follows an hybrid programming fashion with both configurations and native PyTorch scripts. Thus, **scripts** and **configs** folders are created inside bundle. The **scripts** folder contains several PyTorch scripts, which describes logics of model training, inference, validation in fine details. And the **configs** provides configurations (including hyper-parameters, network, loss function, optimizer, data augmentation, etc.) for different operations. The abstract ".yaml" files allow users to quickly modify the algorithm to try different configuration recipes.

```
algorithm_x
    ├── configs
    │   ├── hyper_parameters.yaml
    │   ├── network.yaml
    │   ├── transforms_infer.yaml
    │   ├── transforms_train.yaml
    │   └── transforms_validate.yaml
    ├── models
    │   ├── accuracy_history.csv
    │   ├── best_metric_model.pt
    │   ├── Events
    │   │   └── events.out.tfevents.12345678
    │   └──  progress.yaml
    └── scripts
        ├── __init__.py
        ├── infer.py
        ├── train.py
        └── validate.py
```

In order to run model training of several bundle algorithms in parallel, users can manually utilize the following bash commands inside bundle folder to launch model training within different computing environments (via [Python Fire library](https://github.com/google/python-fire)). After model training is accomplished, the model ensemble can be further conducted with existing bundle algorithms. All commands (including model training, inference, and validation) are described in **README.md** file of each bundle.

```bash
## single-gpu training
python -m scripts.train run --config_file "['configs/hyper_parameters.json','configs/network.yaml','configs/transforms_train.json','configs/transforms_validate.json']"

## multi-gpu (8-gpu) training
torchrun --nnodes=1 --nproc_per_node=8 -m scripts.train run --config_file "['configs/hyper_parameters.json','configs/network.yaml','configs/transforms_train.json','configs/transforms_validate.json']"
```

After the model is trained, all related output files are saved in the **models** folder, including model checkpoint **best_metric_model.pt** (with best validation accuracy), and training history files. Among all training history files, **Events** folders contain event files for learning curve visualization via [TensorBoard](https://www.tensorflow.org/tensorboard). The file **accuracy_history.csv** lists details about the model training progress, including training loss, validation accuracy, number of training steps, and training time cost. It can be used as an alternative of TensorBoard events. The file **progress.yaml** records the best validation accuracy and when it was reached.

### A Quick Way to Try Different Hyper-Parameters

In each bundle algorithm, we provide an easy API for users to quickly update hyper-parameters in model training, inference, and validation. Users can update any parameter at different levels in the configuration ".yaml" file by appending strings to the bash command starting with "--". The values after "--" would override the default values in the configurations. The following command shows a multi-GPU training example. During the actual model training, the learning rate becomes 0.001, **num_images_per_batch** is increased to 6, and the momentum in optimizers is updated to 0.99.

```bash
torchrun --nnodes=1 --nproc_per_node=8 -m scripts.train run --config_file "['configs/hyper_parameters.json','configs/network.yaml','configs/transforms_train.json','configs/transforms_validate.json']  --learning_rate 0.001 --num_images_per_batch 6 --optimizer#momentum 0.99"
```
