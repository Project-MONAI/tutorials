## Hyperparameter Optimization (HPO)
We provide a base HPO generator class `HPOGen` to support the interactions between our Algorithm and third-party
HPO packages like Microsoft Neural Network Intelligence [NNI](https://nni.readthedocs.io/en/stable/). Child classes of HPOGen, including `NNIGen` and `OptunaGen`, are provided to support NNI hyperparameter optimization and Optuna for our algorithms.

> NOTE: We only provide API for interaction between our algorithm and 3rd party HPO packages. The running and configuration of 3rd party packages should be done by users. It can support general HPO algorithms like random search, grid search, and simple bayesian optimization. More advanced HPO algorithms (methods with early stopping, BOHB, PBT, evolution, etc.) may require user modifications to the generated bundle code and the NNIGen/OptunaGen code.

### Concepts
The basic workflow is shown in
<div align="center"> <img src="../figures/hpo_workflow0.png" width="800"/> </div>
<div align="center"> <img src="../figures/hpo_workflow1.png" width="800"/> </div>
The HPOGen class has a `run_algo()` function, which will be used by the third-party HPO packages. `run_algo()` has three steps: hyperparameter sampling (by calling functions provided by 3rd party packages), generates monai bundle folders, and performs training. The validation accuracy will be returned to the third-party package package.

### Usage
The tutorial on how to use NNIGen is [here](../notebooks/hpo_nni.ipynb) and the tutorial for OptunaGen is [here](../notebooks/hpo_optuna.ipynb). The list of HPO algorithms in NNI and Optuna can be found on [the NNI GitHub page](https://github.com/microsoft/nni) and [Optuna documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html).
For demonstration purposes, both of our tutorials use a Grid Search HPO algorithm to optimize the learning rate in training. Users can be easily modified to random search and bayesian based methods for more hyperparameters.
