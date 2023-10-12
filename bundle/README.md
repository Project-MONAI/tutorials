# MONAI Bundle

This directory contains the tutorials and materials for MONAI Bundles. A bundle is a self-describing network which
packages network weights, training/validation/testing scripts, Python code, and ancillary files into a defined
directory structure. Bundles can be downloaded from the model zoo and other sources using MONAI's inbuilt API.
These tutorials start with an introduction on how to construct bundles from scratch, and then go into more depth
on specific features.

All other bundle documentation can be found at https://docs.monai.io/en/latest/bundle_intro.html.

Start the tutorial notebooks on constructing bundles:

1. [Bundle Introduction](./01_bundle_intro.ipynb): create a very simple bundle from scratch.
2. [MedNIST Classification](./02_mednist_classification.ipynb): train a network using the bundle for doing a real task.
3. [MedNIST Classification With Best Practices](./03_mednist_classification_v2.ipynb): do the same again but better.
4. [Integrating Existing Code](./04_integrating_code.ipynb): discussion on how to integrate existing, possible non-MONAI, code into a bundle.

More advanced topics are covered in this directory:

* [Further Features](./further_features.md): covers more advanced features and uses of configs, command usage, and
programmatic use of bundles.

* [introducing_config](./introducing_config): a simple example to introduce the MONAI bundle config and parsing
implementing a standalone program.

*  [customize component](./custom_component): illustrates bringing customized python components, such as transform,
network, and metrics, into a configuration-based workflow.

*  [hybrid programming](./hybrid_programming): shows how to parse the config files in your own python program,
instantiate necessary components with python program and execute the inference.

* [python bundle workflow](./python_bundle_workflow): step-by-step tutorial examples show how to develop a bundle
training or inference workflow in Python instead of JSON / YAML configs.

Other tutorials and resources within MONAI on other bundle topics can be found here:

* [MONAI Deploy App SDK](https://github.com/Project-MONAI/monai-deploy-app-sdk/tree/main/notebooks/tutorials) including bundle tutorials
* [MONAI Label Bundle App](https://github.com/Project-MONAI/MONAILabel/tree/main/sample-apps/monaibundle)
