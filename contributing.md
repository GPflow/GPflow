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

With GPflow, we aim to make an extensible library for Gaussian processes which makes building complex models easy. In order to do this, we aim to make GPflow a complete library for doing inference and prediction in sophisticated ways (focussing mainly on variational inference) for *single layer models* only. In order to allow more complicated models to be implemented, we also provide functionality for latent / uncertain inputs. Other models, like deep GPs, can be implemented in their own repository by using GPflow as a dependency. We choose to limit the scope deliberately in order to ensure a high quality codebase.

We welcome contributions to GPflow. If you would like to contribute a feature, please raise discussion via a GitHub issue, to discuss the suitability of the feature within GPflow. If the feature is outside the envisaged scope, we can still link to a separate project in our Readme. Large features also make it onto the [roadmap](roadmap.md).

### I have this big feature/extension I would like to add...

Due to limited scope we may not be able to review and merge every feature, however useful it may be. Particularly large contributions or changes to core code are harder to justify against the scope of the project or future development plans. For such contributions, we suggest you publish them as a separate package that extends GPflow. We can link to your project from an issue discussing the topic or within the repository. Discussing a possible contribution in an issue should give an indication to how broadly it is supported to bring it into the codebase.

### ...but it won't work without changes to GPflow core?

We aim to have the GPflow core infrastructure be sufficiently extensible and modular to enable a wide range of third-party extensions without having to touch the core of GPflow. The `features` module is an example of this, to enable multiscale inducing features, Fourier features, etc. If your feature/extension does not work outside of GPflow-core because something is hard-coded, please open an issue to discuss this with us!

## Code quality requirements

- Code must be covered by tests. We strongly encourage you to use the [pytest](https://docs.pytest.org/) framework. Even when you see your tests as a part of the old-fashioned GPflow [test cases](https://docs.python.org/3/library/unittest.html) it is still recommended to write a new test or modify the old one to use `pytest`.
- The code must be documented. We use *reST* in docstrings. *reST* is a [standard way of documenting](http://docs.python-guide.org/en/latest/writing/documentation/) in python.\
If the code which you are working on does not yet have any documentation, we would be very grateful if you could amend the deficiency. Missing documentation leads to ambiguities and difficulties in understanding future contributions and use cases.
- Use [type annotations](https://docs.python.org/3/library/typing.html). Type hints make code cleaner and _safer_ to some extent.
- Python code should follow the *PEP8* style. Use `pylint` and `mypy` for formatting and _type checking_. GPflow project has `.pylintrc` with some relaxed naming conventions.
- Practise writing good code as far as is reasonable. Simpler is usually better. Reading the existing GPflow code should give a good idea of the expected style.

Example:

```python
class Foo:
    """This is an example class with a single simple static method.
    It mimics a singleton class which can run TensorFlow tensors."""

    @classmethod
    def add_one(cls, tensor: tf.Tensor, op_name: Optional[str] = None) -> Union[int, float, np.ndarray]:
        """Increment input tensor and run it in default GPflow session.
        It is assumed that you have already instantiated variables in 
        the default GPflow session before calling this method.

        :param tf.Tensor tensor: Input tensor.
        :param str op_name: Name scope for TensorFlow operation.

        :return: Result depends on which value was passed to
            the method. It can be either a scalar or NumPy array.
        """
        default_name = self.__class__.__name__
        name = op_name if op_name is not None else default_name
        session = gpflow.get_default_session()
        with tf.name_scope(name):
            incr_tensor = tensor + 1
        return session.run(incr_tensor)
```

## Pull requests and the master branch

If you think that your contribution falls within the project scope (see above) please submit a Pull Request (PR) to our GitHub page. In order to maintain code quality, and make life easy for the reviewers, please ensure that your PR:

- Only fixes one issue or adds one feature.
- Makes the minimal amount of changes to the existing codebase.
- Minimises the amount of changes to IPython notebooks (i.e. please do not commit notebooks which are simply re-run).

All code that is destined for the master branch of GPflow goes through a PR. Only a small number of people can merge PRs onto the master branch (currently [Artem Artemev](https://github.com/awav), [James Hensman](https://github.com/jameshensman), [Alex Matthews](https://github.com/alexggmatthews), [Mark van der Wilk](https://github.com/markvdw) and [Alexis Boukouvalas](https://github.com/alexisboukouvalas)).


## Tests and continuous integration

GPflow is ~99% covered by the testing suite. We expect changes to code to pass these tests, and for new code to be covered by new tests. Currently, tests are run by travis and coverage is reported by codecov.

## Documentation

GPflow's documentation is not comprehensive, but covers enough to get users started. We expect that new features have documentation that can help others get up to speed. The docs are mostly IPython notebooks that compile into HTML via Sphinx, using nbsphinx.

## Version numbering

The main purpose of versioning GPflow is user convenience.

We use the [semantic versioning scheme](https://semver.org/). The semver implies `MAJOR.MINOR.PATCH` version scheme, where `MAJOR` changes when there are incompatibilities in API, `MINOR` means adding functionality without breaking existing API and `PATCH` presumes the code update has backward compatible bug fixes.

When incrementing the version number, the following tasks are required:

- Update the version in `gpflow/_version.py`
- Update the version in the `doc/source/conf.py`
- Add a note to `RELEASE.md`

### Keeping up with TensorFlow

GPflow tries to keep up with API changes in TensorFlow as far as is reasonable, so that the latest GPflow will work with the latest stable TensorFlow release. Changing the version of TensorFlow that we're compatible with requires a few tasks:

- Update version used on travis via `travis.yml`
- Update version used on codeship (requires codeship login)
- Update `README.md`
- Update version used by readthedocs.org via `docsrequire.txt`
- Increment the GPflow version (see below).
