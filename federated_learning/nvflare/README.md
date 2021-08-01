**Federated learning with [NVFlare](./federated_learning/nvflare)**

The examples here show how to train federated learning models with [NVFlare](https://pypi.org/project/nvflare/) and MONAI-based trainers.

1. [nvflare_example](./nvflare_example/README.md) shows how to run NVFlare with MONAI on a local machine to simulate an FL setting (server and client communicate over localhost). It also shows how to run a simulated FL experiment completely automated using the admin API. To streamline the experimentation, we have already prepared startup kits for up to 8 clients in this tutorial.
   
2. [nvflare_example_docker](./nvflare_example/README.md) provides further details on running FL with MONAI and NVFlare using docker containers for the server and each client for easier real-world deployment.




