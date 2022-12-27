- [Introduction](#introduction)
- [The contribution process](#the-contribution-process)
  - [Raise an issue](#raise-an-issue)
  - [Create a fork](#create-a-fork)
  - [Create a new folder](#create-a-new-folder)
  - [Add license](#add-license)
  - [Create a notebook](#create-a-notebook)
  - [Commit new changes](#commit-new-changes)
  - [Open a pull request](#open-a-pull-request)
- [Strong recommendations](#strong-recommendations)
- [CI test passing guide](#ci-test-passing-guide)
- [Benchmarking result report](#benchmarking-result-report)

## Introduction

Thanks for considering a contribution to the MONAI Tutorials Repository and reading the Contributing Guidelines.

The following is a set of guidelines for contributing tutorials to Project MONAI.
The guidelines are based on the discussions in #1119
Please feel free to propose changes to this document in a pull request.

## The contribution process

### Raise an issue

We encourage you to [raise an issue](https://github.com/Project-MONAI/tutorials/issues/new/choose) about your needs for MONAI tutorials.
The idea may generate echo in the community and interest other contributors as well.

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

Finally, the MONAI tutorial has a [README file](README.md) to communicate the important information and provide an overview of the tutorials and examples. For new tutorials, please add a new entry to the [List](README.md#4-list-of-notebooks-and-examples) of notebooks and examples](README.md#4-list-of-notebooks-and-examples).

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
- A markdown cell with `## Setup environment` and a code cell that executes `pip install` shell commands with the exclamation mark`!`` to install all the packages necessary for your tutorial.

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

The tutorial provides [useful templates](.github/contributing_templates/notebook/README.md) for contributors to start with.

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
When the pull request is opened, the MONAI repository has a set of GitHub actions that will run linters and check on the changes.

Please check more details in the [guidelines](#ci-test-passing-guide) on how to pass the tests](#ci-test-passing-guide).

In addition, the team will perform diligent code reviews following this [set of guidelines](#strong-recommendations) to reduce the amount of work for users to run the tutorials.

## Strong recommendations

## CI test passing guide

## Benchmarking result report
