# Contributing to GPflow
This file contains notes for potential contribtors to GPflow, as well as some notes that may be helpful for maintainance.

## Project scope
We do welcome contributions to GPflow. However, the project is deliberately of limited scope, to try to ensure a high quality codebase: if you'd like to contribute a feature, please raise discussion via a github issue. 

## Code Style
 - Python code should follow the pep8 style. To help with this, we suggest using a plugin for your editor. 
 - Practise good code as far as is reasonable. The Google [python syle guide](https://google.github.io/styleguide/pyguide.html) is a good place to look for more information, as is (we hope!) reading the existing GPflow code. 

## Pull requests and the master branch
All code that is destined for the master branch of GPflow goes through a PR. Only a small number of people can merge PRs onto the master branch (currently James Hensman, Alex Matthews, Mark van der Wilk and Alexis Boukouvalas). 

## Tests and continuous integration
GPflow is 99% covered by the testing suite. We expect changes to code to pass these tests, and for new code to be covered by new tests. Currently, tests are run by travis (python 3) and by codeship (python 2.7), coverage is reported by codecov. 

## Python 2 and 3
GPflow aims to work in both python 2.7 and 3.5. Tests should pass in both. 

## Documentation
GPflow's documentation is not comprehensive, but covers enough to get users started. We expect that new features have documentation that can help other get up to speed. The docs are mostly IPython notebooks that compile into html via sphinx, using nbsphinx.

## Keeping up with tensorflow
GPflow tries to keep up with api changes in tensorflow as far as is reasonable, so that the latest GPflow will work with the latest (stable) tensorflow. Changing the version of tensorflow that we're compatible with requires a few tasks:
 - update version used on travis via `travis.yml`
 - update version used on codeship (requires codeship login)
 - update `README.md`
 - update version ussed by readthedocs.org via `docsrequire.txt`

## Version numbering
The main purpose of versioning GPflow is user convenience. 
When incrementing the version number, the following tasks are required:
 - Update the version in `GPflow/_version.py`
 - Udate the version in the `doc/source/conf.py`
 - Add a note to `RELEASE.md`
