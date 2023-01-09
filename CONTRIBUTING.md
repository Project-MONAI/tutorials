- [Introduction](#introduction)
- [The contribution process](#the-contribution-process)
  - [Raise an issue](#raise-an-issue)
  - [Create a fork](#create-a-fork)
  - [Create a new folder](#create-a-new-folder)
  - [Add license](#add-license)
  - [Create a notebook](#create-a-notebook)
  - [Commit new changes](#commit-new-changes)
  - [Open a pull request](#open-a-pull-request)
  - [Common recommendations for the review](#common-recommendations-for-the-review)
- [CI/CD test passing guide](#cicd-test-passing-guide)
  - [PEP 8 Style](#pep-8-style)
  - [Notebook execution](#notebook-execution)
  - [Format requirements](#format-requirements)
- [Benchmarking result report](#benchmarking-result-report)

## Introduction

Thank you for considering a contribution to the MONAI Tutorials Repository and reading the Contributing Guidelines.

The following is a set of guidelines for contributing tutorials to Project MONAI.
We want the guidelines to support the growth of the project, instead of restricting the depth and breadth of your contribution.
If you have any questions about the guideline, please communicate with us and we are happy to discuss them with you.
MONAI is an open-source project and its success relies on the entire community.

Please feel free to propose changes to this document in a pull request (PR).

## The contribution process

### Raise an issue

We encourage you to [raise an issue](https://github.com/Project-MONAI/tutorials/issues/new/choose) about your needs for MONAI tutorials.
Conversations help better define the feature and allow other contributors in the community to provide feedback.

### Create a fork

When you are ready to kick off some coding, [create a new fork](https://github.com/Project-MONAI/tutorials/fork) of the `main` branch of this repository.

After forking is complete, you can `git clone` the forked repo to your development environment, and use `git checkout -b` to create a new branch to freely experiment with your changes.

For example:
```bash
git clone https://github.com/<your github space>/tutorials.git
git checkout -b <your new branch name>
```

### Create a new folder

MONAI tutorials covered examples of various medical applications and technical topics.
Tutorials under the same topic/application share the same folder under the root directory, e.g. `2d_classification`, `2d_segmentation`, `3d_classification`, and `3d_segmentation`.

To hold your work in a new location, you can choose to create a new subfolder under the existing ones, or a folder under the root directory if it doesn't belong to any of the existing topics/applications.

The folder should have a `README.md` file with descriptive information for others to start using your code or tutorial notebooks.

Finally, the MONAI tutorial has a [README file](README.md) to communicate the important information and provide an overview of the tutorials and examples. For new tutorials, please add a new entry to the [List](README.md#4-list-of-notebooks-and-examples) of notebooks and [examples](README.md#4-list-of-notebooks-and-examples).

### Add license

All source code and notebook files should include copyright information at the top of the file.

NOTE: for Jupyter Notebook `.ipynb` files, the copyright information should appear at the top of the first markdown cell.
There are extra two spaces at the end of each line to ensure no line auto-wrap in the markdown rendering in the display.

```markdown
Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

```

### Create a notebook

Jupyter Notebook is the preferred way of writing a new MONAI tutorial because we encourage contributors to visualize the outputs and keep the records with the code.

Writing a notebook is easy and flexible, but we require all tutorial notebooks to start with the following three sections:

- Title of the notebook with [licensing information](#add-license) in one markdown cell
- A markdown cell with `## Setup environment` and a code cell that executes `pip install` shell commands with the exclamation mark `!` to install all the packages necessary for your tutorial.

    For example, we install [MONAI with extra dependencies](https://docs.monai.io/en/stable/installation.html#installing-the-recommended-dependencies) and `matplotlib` to set up the environment for the [mednist_tutorial](./2d_classification/mednist_tutorial.ipynb):

    ```
    !python -c "import monai" || pip install -q "monai-weekly[pillow, tqdm]"
    !python -c "import matplotlib" || pip install -q matplotlib
    ```

- A markdown cell with `## Setup imports` and a code cell that contains **ALL** import statements and ends with `monai.config.print_config()`:

    ```python
    import numpy as np

    from monai.config import print_config
    from monai.data import DataLoader
    ...

    print_config()
    ```

Following this guideline, we prepare some [templates](.github/contributing_templates/notebook/README.md) for contributors to start with.

### Commit new changes

MONAI enforces the [Developer Certificate of Origin](https://developercertificate.org/) (DCO) on all pull requests.
All commit messages should contain the `Signed-off-by` line with an email address.
The [GitHub DCO app](https://github.com/apps/dco) is deployed on MONAI.
The pull request's status will be `failed` if commits do not contain a valid `Signed-off-by` line.

Git has a `-s` (or `--signoff`) command-line option to append this automatically to your commit message:
```bash
git commit -s -m 'your awesome commit summary'
```
The commit message will be:
```
    your awesome commit summary

    Signed-off-by: Your Name <yourname@example.org>
```

Full text of the DCO:
```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

### Open a pull request

Once the changes are ready, push your local branch to the remote, e.g. `git push --set-upstream origin <your new branch name>`, and [open a pull request](https://github.com/Project-MONAI/tutorials/pulls).
When the pull request is opened, the MONAI repository has a set of GitHub actions that will run checks on the changes.

Please check more [details in the guidelines](#ci-test-passing-guide) for how to pass the tests.

In addition, the team will perform diligent code reviews following this [set of guidelines](#common-recommendations-for-the-review) to reduce the amount of work for users to run the tutorials.

### Common recommendations for the review

Here are some recommendations to make your pull requests faster to review:
- Note dataset availability. The contributor needs to provide info on how to access the dataset and make a note about the dataset's licensing info. For example, some datasets may be used for non-commercial purposes only. If the dataset can be directly downloaded from a public source on the internet, please consider using the folder specified by `MONAI_DATA_DIRECTORY` to store the dataset files as the [example notebook](.github/contributing_templates/notebook/example_feature.ipynb) shows.

- Avoid large files. Dataset files should be removed from the PR contribution. The overall size of the notebook should be kept down to a few megabytes.
- Clean up long text outputs generated by Jupyter notebook code cells.
- Remove private information from the notebook. For example, the user name in the file paths in the notebook outputs and metadata.
- Be aware of the Hyperlink usage in the notebook:
  - Avoid linking MONAI tutorial resources in the repo using web links (instead, use relative file paths)
  - Avoid linking folders (folder links do not work well in Jupyter notebooks)
  - For graphs, it is recommended to download them and add them to the repo in the `./figure` folder under the root directory

If your tutorial includes network training pipelines, we encourage implementations to scale the training on multiple GPUs.
For example, if you are using `torchrun` for multi-GPU training, please feel free to include the Python scripts in your tutorial.

## CI/CD test passing guide

The testing system uses `papermill` to run the notebooks.
To verify the tutorial notebook locally, you can `pip install jupytext flake8 papermill` and then issue the following command with the full path to the notebook file.

```
./runner.sh -t <path to your .ipynb file>
```

NOTE: the argument after `-t` provides a filename for the `runner.sh` to locate a single notebook in the tutorial to run tests on.
It is equivalent to using a regex pattern in the argument `-p` or `--pattern` to search for files to run checks.
In this case, we can also use `-wholename` to specify the only notebook file we would like to check.
The path must begin with `./`, for example:
```
./runner.sh -p "-and -wholename './2d_classification/mednist_tutorial.ipynb'"
```

If you have multiple notebooks to test, please consider designing a regex pattern to locate all the notebook files and use `-p`.

The `runner.sh` includes three kinds of checks: PEP 8 Style, notebook execution, and format requirement.

### PEP 8 Style

PEP 8 is a set of guidelines for writing Python code.
It was created to improve the readability and consistency of Python code, and to make it easier for people to understand and maintain.

The guidelines cover a wide range of topics, including naming conventions, indentation, white space, and comments.
They also include recommendations for how to structure and format your code, as well as how to write comments and documentation.

PEP8 style is required for all Python code in `.py` script files and the cell blocks in Jupyter notebooks for the MONAI tutorial.
Here is a set of common mistakes that lead to check failures:
- A blank line at end of the cell
- Extra blank spaces at the end of some lines
- Module import is not in the `Setup import` cell
- Import an unused module or create an unused variable
It needs to note that `--autofix` needs a few additional packages to help you fix some of the issues automatically, and most others need your manual correction.

To run the PEP 8 tests locally, you can use this argument `--no-run` to run the scan only:

```
./runner.sh -t <path to your .ipynb file> --no-run
```

### Notebook execution

The CI/CD in the pull request process does not execute a Jupyter notebook.
But the MONAI tutorial has scheduled scans to check if all notebooks can be executed regularly.

The notebook must be in a self-contained state, e.g. setting up the Python environment, downloading the dataset, performing the analysis, and preferably, cleaning up the intermediate files generated from the execution.

During integration testing, we run these notebooks.
To save time, we modify variables to avoid unnecessary `for` loop iterations.
For example, the testing system will search for `max_epoch` in the notebook, and set to value to `1` for faster notebook testing.

Hence, during training please use the variables:
- `max_epochs` for the number of training epochs
- `val_interval` for the validation interval

On the other hand, if the training is not part of your tutorial, or doesn't use the idea of epochs, please update the exclusion list of `doesnt_contain_max_epochs` in the [runner.sh](runner.sh).
This lets the runner know that it's not a problem if it doesn't find `max_epochs`.

If you have any other variables that would benefit from setting them to `1` during testing, add them to `strings_to_replace` in `runner.sh`.
These variables have been added to the list by other contributors:
- `disc_train_interval` for GAN discriminator training inteval
- `disc_train_steps` for GAN discriminator training steps
- `num_batches_for_histogram`

Finally, if your tutorial is not suitable for automated testing, please exclude the notebook by updating the `skip_run_papermill` in the [runner.sh](runner.sh).
You can append another line in the `skip_run_papermill`:
```
skip_run_papermill=("${skip_run_papermill[@]}" .*<name of your notebook>*)
```

If `runner.sh` is modified in your PR, the file permission of `runner.sh` could be set incorrectly in some cases.
Please ensure the file permission is set to `-rwxrwxr-x` so that our test system can load the script.
For more information about how to change file permission in git version control, this [StackOverflow page](https://stackoverflow.com/questions/10516201/updating-and-committing-only-a-files-permissions-using-git-version-control) could be helpful.

### Format requirements

The CI/CD will check the following formats, in addition to PEP 8:
- [Licensing information](#add-license)
- [Environment and imports](#create-a-notebook)
- [Output text length](#common-recommendations-for-the-review)

## Benchmarking result report

To standardize all result reporting, the MONAI team recommends contributors use A100 as the standard device for benchmarking in all notebooks.
If contributors have difficulties getting the computation resources, please contact our team for support.
