## Hyperparameter Optimization (HPO)
We provide a base HPO generator class (HPOGen) to support interactions between our Algorithm and 3rd party
HPO packages like Microsoft Neural Network Intelligence [NNI](https://nni.readthedocs.io/en/stable/). Child classes of HPOGen: NNIGen and OptunaGen, are provided to support NNI hyperparameter optimization and Optuna for our algorithm. Tutorial on how to use NNIGen is [here](../notebooks/hpo.ipynb) and OptunaGen [here](https://optuna.readthedocs.io/en/stable/). The list of HPO algorithms in NNI and Optuna can be found [here](https://github.com/microsoft/nni) and [here](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html).

NOTE: We only provide API for interaction between our algorithm and 3rd party HPO packages. The running and configuration of 3rd party packages should be done by users. It can supports general HPO algorithms like random search, grid search and simple bayesian optimization. More advanced HPO algorithms (methods with early stopping, BOHB, PBT, evolution e.t.c) may require user modifications to the generated bundle code and the NNIGen/OptunaGen code.

### Concepts
The basic workflow is shown in
<div align="center"> <img src="../figures/hpo_workflow0.png" width="800"/> </div>
<div align="center"> <img src="../figures/hpo_workflow1.png" width="800"/> </div>
The HPOGen class has a run_algo() function, which will be used by 3rd party HPO packages. run_algo() does three steps: hyperparameter sampling (by calling functions provided by 3rd party packages), generates monai bundle folders, and performs training. The validation accuracy will be returned to the 3rd party package program.

### Usage
A full NNI tutorial is [here](../notebooks/hpo_nni.ipynb). The Optuna tutorial is [here](../notebooks/hpo_optuna.ipynb). They are based on Grid search for learning rate but can be easily modified to random search and bayesian based methods for more hyperparameters.
