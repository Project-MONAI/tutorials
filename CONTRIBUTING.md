- [Introduction](#introduction)
- [The contribution process](#the-contribution-process)
- [Hard requirements](#hard-requirements)
- [Strong recommendations](#strong-recommendations)
- [CI test passing guide](#ci-test-passing-guide)
- [Benchmarking result report](#benchmarking-result-report)

## Introduction

Thanks for considering a contribution to the MONAI Tutorials Repository and reading the Contributing Guidelines.

The following is a set of guideline for contributing tutorials to the Project MONAI.
The guidelines are based on the discussions in #1119
Please feel free to propose changes to this document in a pull request.

## The contribution process

### Raise an issue

We encourage you to [raise an issue](https://github.com/Project-MONAI/tutorials/issues/new/choose) about your needs for MONAI tutorials.
The idea may generate echo in the community and interest other contributors as well.

### Fork this repository

When you are ready to kick off some coding, [create a new fork](https://github.com/Project-MONAI/tutorials/fork) of the `main` branch of this repository.

After forking is complete, you can `git clone` the forked repo to your development environment, and use `git checkout -b` to create a new branch to freely experiment your changes.

For example:
```bash
git clone https://github.com/<your github space>/tutorials.git
git checkout -b <your new branch name>
```

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
As soon as the pull request is opened, the MONAI repository has a set of Github actions that will run linters and checks on the changes.

Please check more details about [the hard requirements](#hard-requirements) and [guidelines how to pass the tests](#ci-test-passing-guide).

In addition, the team will perform diligent code reviews following this [set of guidelines](#strong-recommendations) to reduce the amount of work for users to run the tutorials.

## Hard requirements

## Strong recommendations

## CI test passing guide

## Benchmarking result report
