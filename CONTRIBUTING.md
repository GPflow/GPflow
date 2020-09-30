# Contributing to GPflow

This file contains notes for potential contributors to GPflow, as well as some notes that may be helpful for maintenance.

#### Table Of Contents

* [Project scope](#project-scope)
* [Code quality requirements](#code-quality-requirements)
* [Pull requests and the master branch](#pull-requests-and-the-master-branch)
* [Tests and continuous integration](#tests-and-continuous-integration)
* [Documentation](#documentation)
* [Version numbering](#version-numbering)
    * [Keeping up with TensorFlow](#keeping-up-with-tensorflow)


## Project scope

With GPflow, we aim to make an extensible library for Gaussian processes which makes building complex models easy. In order to do this, we aim to make GPflow a complete library for doing inference and prediction in sophisticated ways (focussing mainly on variational inference) for *single layer models* only. In order to allow more complicated models to be implemented, we also provide functionality for latent / uncertain inputs. Other models, like deep GPs, can be implemented in their own repository by using GPflow as a dependency. We choose to limit the scope deliberately in order to ensure a high-quality codebase.

We welcome contributions to GPflow. If you would like to contribute a feature, please raise discussion via a GitHub issue, to discuss the suitability of the feature within GPflow. If the feature is outside the envisaged scope, we can still link to a separate project in our Readme.

### I have this big feature/extension I would like to add...

Due to limited scope we may not be able to review and merge every feature, however useful it may be. Particularly large contributions or changes to core code are harder to justify against the scope of the project or future development plans. For such contributions, we suggest you publish them as a separate package that extends GPflow. We can link to your project from an issue discussing the topic or within the repository. Discussing a possible contribution in an issue should give an indication to how broadly it is supported to bring it into the codebase.

### ...but it won't work without changes to GPflow core?

We aim to have the GPflow core infrastructure be sufficiently extensible and modular to enable a wide range of third-party extensions without having to touch the core of GPflow. The `inducing_variables` module is an example of this to enable interdomain approximations (multiscale inducing features, Fourier features, etc.). If your feature/extension does not work outside of GPflow-core because something is hard-coded, please open an issue to discuss this with us! We are happy to discuss and implement changes to the core code that make it easier for you to extend GPflow with a separate package.

## Code quality requirements

- Code must be covered by tests. We strongly encourage you to use the [pytest](https://docs.pytest.org/) framework.
- The code must be documented. We use *reST* in docstrings. *reST* is a [standard way of documenting](http://docs.python-guide.org/en/latest/writing/documentation/) in python.\
If the code which you are working on does not yet have any documentation, we would be very grateful if you could amend the deficiency. Missing documentation leads to ambiguities and difficulties in understanding future contributions and use cases.
- Use [type annotations](https://docs.python.org/3/library/typing.html). Type hints make code cleaner and _safer_ to some extent.
- Python code should generally follow the *PEP8* style. We use some custom naming conventions (see below) to have our notation follow the Gaussian process literature. Use `pylint` and `mypy` for formatting and _type checking_. GPflow project has a `.pylintrc` with some relaxed naming conventions.
- Practise writing good code as far as is reasonable. Simpler is usually better. Reading the existing GPflow code should give a good idea of the expected style.

### Naming conventions

Variable names: scalars and vectors start lowercase, but following the notation used in Gaussian process papers, all matrices are denoted with upper case. For example, `lengthscales` denotes a vector (i.e., rank 1, for example a tensor with shape [D]), whereas `Xnew` denotes a matrix (i.e., rank 2, for example a tensor with shape [N, D]; note that a [N, 1] tensor is a matrix, not a vector).

### Formatting

GPflow uses [black](https://github.com/psf/black) and [isort](https://pycqa.github.io/isort/) for formatting. Simply run `make format` from the GPflow root directory (or check our Makefile for the appropriate command-line options).

## Pull requests

If you think that your contribution falls within the project scope (see above) please submit a Pull Request (PR) to our GitHub page.
(GitHub provides extensive documentation on [forking](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) and [pull requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests).)

In order to maintain code quality, and make life easy for the reviewers, please ensure that your PR:

- Only fixes one issue or adds one feature.
- Makes the minimal amount of changes to the existing codebase.
- Is testing its changes.
- Passes all checks (formatting, types, tests - you can run them all locally using `make check-all` from the GPflow root directory).

All code goes through a PR; there are no direct commits to the master and develop branches.

## Tests and continuous integration

GPflow is ~97% covered by the testing suite. We expect changes to code to pass these tests, and for new code to be covered by new tests. Currently, tests are run by CircleCI and coverage is reported by codecov. Pull requests should aim to have >97% *patch* coverage (i.e., all the lines that are *changing* should be covered by tests).

## Documentation

GPflow's documentation is not comprehensive, but covers enough to get users started. We expect that new features have documentation that can help others get up to speed. The docs are mostly IPython notebooks (stored in the git repository in Jupytext format) that compile into HTML via Sphinx, using nbsphinx.

## Version numbering

The main purpose of versioning GPflow is user convenience.

We use the [semantic versioning scheme](https://semver.org/). The semver implies `MAJOR.MINOR.PATCH` version scheme, where `MAJOR` changes when there are incompatibilities in API, `MINOR` means adding functionality without breaking existing API and `PATCH` presumes the code update has backward compatible bug fixes.

When incrementing the version number, this has to be reflected both in `./VERSION` and in `./doc/source/conf.py`.

### Keeping up with TensorFlow

GPflow tries to keep up with API changes in TensorFlow as far as is reasonable, so that the latest GPflow will work with the latest stable TensorFlow release. Changing the version of TensorFlow that we're compatible with requires a few tasks:

- Update `README.md`
- Increment the GPflow version (see above).
