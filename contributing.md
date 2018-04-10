# Contributing to GPflow
This file contains notes for potential contributors to GPflow, as well as some notes that may be helpful for maintenance.

## Project scope
We do welcome contributions to GPflow. However, the project is deliberately of limited scope, to try to ensure a high quality codebase: if you'd like to contribute a feature, please raise discussion via a GitHub issue. Large features also make it onto the [roadmap](roadmap.md).

Due to limited scope we may not be able to review and merge every feature, however useful it may be. Particularly large contributions or changes to core code are harder to justify against the scope of the project or future development plans. For such contributions, we suggest you publish them as a separate package that extends GPflow. We can link to your project from an issue discussing the topic or within the repository. Discussing a possible contribution in an issue should give an indication to how broadly it is supported to bring it into the codebase.

## Code Style
 - Python code should follow the pep8 style. To help with this, we suggest using a plugin for your editor.
 - Practise writing good code as far as is reasonable. Simpler is usually better. Complicated language features (I'm looking at you, metaclasses) are out. Reading the existing GPflow code should give a good idea of the expected style.

## Pull requests and the master branch
If you think that your contribution falls within the project scope (see above) please submit a Pull Request (PR) to our GitHub page. In order to maintain code quality, and make life easy for the reviewers, please ensure that your PR:
- Only fixes one issue or adds one feature.
- Makes the minimal amount of changes to the existing codebase.
- Minimises the amount of changes to IPython notebooks (i.e. please do not commit notebooks which are simply re-run).

All code that is destined for the master branch of GPflow goes through a PR. Only a small number of people can merge PRs onto the master branch (currently James Hensman, Alex Matthews, Mark van der Wilk and Alexis Boukouvalas).

## Tests and continuous integration
GPflow is 99% covered by the testing suite. We expect changes to code to pass these tests, and for new code to be covered by new tests. Currently, tests are run by travis (python 3) and by codeship (python 2.7), coverage is reported by codecov.

To save time during development, slow tests are marked with a 'speed' attribute. To run the tests without the slow ones, use `nosetests -A "speed!='slow'" testing`. By default, all tests are run, including on travis/codeship.

## Python 3
GPflow supports python >= 3.5

## Documentation
GPflow's documentation is not comprehensive, but covers enough to get users started. We expect that new features have documentation that can help others get up to speed. The docs are mostly IPython notebooks that compile into html via sphinx, using nbsphinx.

## Keeping up with TensorFlow
GPflow tries to keep up with API changes in TensorFlow as far as is reasonable, so that the latest GPflow will work with the latest (stable) TensorFlow. Changing the version of TensorFlow that we're compatible with requires a few tasks:
 - Update version used on travis via `travis.yml`
 - Update version used on codeship (requires codeship login)
 - Update `README.md`
 - Update version used by readthedocs.org via `docsrequire.txt`
 - Increment the GPflow version (see below).

## Version numbering
The main purpose of versioning GPflow is user convenience: to keep the number of releases down, we try to combine several PRs into one increment. As we work towards something that we might call 1.0, minor version bumps (X.1.X) are reserved for changes that alter the underlying code or code structure significantly. Minor-minor version bumps (X.X.1) are used for changes that change the GPflow API, update to follow a new TensorFlow API, or introduce incremental new features.
When incrementing the version number, the following tasks are required:
 - Update the version in `gpflow/_version.py`
 - Update the version in the `doc/source/conf.py`
 - Add a note to `RELEASE.md`
