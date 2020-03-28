<div style="text-align:center">
<img width="500" height="200" src="./doc/source/_static/gpflow_logo.svg">
</div>

[![CircleCI](https://circleci.com/gh/GPflow/GPflow/tree/develop.svg?style=svg)](https://circleci.com/gh/GPflow/GPflow/tree/develop)
[![Coverage Status](http://codecov.io/github/GPflow/GPflow/coverage.svg?branch=master)](http://codecov.io/github/GPflow/GPflow?branch=master)
[![Documentation Status](https://readthedocs.org/projects/gpflow/badge/?version=master)](http://gpflow.readthedocs.io/en/master/?badge=master)
[![Slack Status](https://img.shields.io/badge/slack-gpflow-green.svg?logo=Slack)](https://join.slack.com/t/gpflow/shared_invite/enQtOTE5MDA0Nzg5NjA2LTYwZWI3MzhjYjNlZWI1MWExYzZjMGNhOWIwZWMzMGY0YjVkYzAyYjQ4NjgzNDUyZTgyNzcwYjAyY2QzMWRmYjE)


[Website](https://gpflow.org) |
[Documentation (develop/v2.0)](https://gpflow.readthedocs.io/en/develop/) |
[Documentation (v1.5)](https://gpflow.readthedocs.io/en/v1.5.1-docs/) |
[Glossary](GLOSSARY.md)

GPflow is a package for building Gaussian process models in Python, using [TensorFlow](http://www.tensorflow.org). It was originally created by [James Hensman](http://jameshensman.github.io/) and [Alexander G. de G. Matthews](https://github.com/alexggmatthews). It is now actively maintained by (in alphabetical order) [Alexis Boukouvalas](https://github.com/alexisboukouvalas), [Artem Artemev](http://github.com/awav/), [Eric Hambro](erichambro.com), [James Hensman](http://jameshensman.github.io/), [Joel Berkeley](https://github.com/joelberkeley), [Mark van der Wilk](https://markvdw.github.io/), [ST John](https://github.com/st--) and [Vincent Dutordoir](https://vdutor.github.io/).
GPflow would not be the same without the community. **We are grateful to [all contributors](http://github.com/GPflow/GPflow/graphs/contributors) who have helped shape GPflow.**
 
 *GPflow is an open source project. If you have relevant skills and are interested in contributing then please do contact us (see ["The GPflow community" section](#the-gpflow-community) below).*


## What does GPflow do?

GPflow implements modern Gaussian process inference for composable kernels and likelihoods. The [online documentation (develop)](http://gpflow.readthedocs.io/en/develop/)/[(master)](http://gpflow.readthedocs.io/en/master/) contains more details.

GPflow 2.0 uses [TensorFlow 2.0](http://www.tensorflow.org) for running computations, which allows fast execution on GPUs, and uses Python ≥ 3.6.


### Install GPflow 2

#### Latest release from PyPI

```bash
pip install gpflow
```

#### Latest source from GitHub

(Be aware that this changes regularly and new commits on the `develop` branch may break your code.)

With the release of _TensorFlow_ 2.1 and _Tensorflow Probability_ 0.9, you should only need to run
```bash
pip install -e .
```
in a check-out of the `develop` branch of the GPflow github repository.

Alternatively, you can use `pip`:

Latest GitHub `develop` version:
```bash
pip install git+https://github.com/GPflow/GPflow.git@develop#egg=gpflow
```
This will automatically install all required dependencies. (Note: as of January 2020, PyPI only has tensorflow 2.x for python 3.6 and 3.7, not for python 3.8, which is the latest release and the Anaconda default.)


### Install GPflow 1.5.1 (last official release)

Using `pip`:
```bash
pip install 'gpflow<2.0' 'tensorflow<2.0'
```
GPflow leaves installing tensorflow to the user so that you can choose whether you want the CPU-only or the GPU-support package. Replace `tensorflow<2.0` with `tensorflow-gpu<2.0` for the GPU version.


## Getting Started with GPflow 2.0

There is an ["Intro to GPflow 2.0"](https://gpflow.readthedocs.io/en/develop/notebooks/intro_to_gpflow2.html) Jupyter notebook; check it out for details. To convert your code from GPflow 1 check the [GPflow 2 upgrade guide](https://gpflow.readthedocs.io/en/develop/notebooks/gpflow2_upgrade_guide.html).



- **GPflow 1.0**

  *We have stopped development and support for GPflow based on TensorFlow 1.0. We now accept only bug fixes to GPflow 1.0 in the **develop-1.0** branch. The latest available release is [v1.5.1](https://github.com/GPflow/GPflow/releases/tag/v1.5.1). [Documentation](https://gpflow.readthedocs.io/en/v1.5.1-docs/) and [tutorials](https://nbviewer.jupyter.org/github/GPflow/GPflow/blob/v1.5.1/doc/source/notebooks/intro.ipynb) will remain available.*


## The GPflow Community

### Getting help

**Bugs, feature requests, pain points, annoying design quirks, etc:**
Please use [GitHub issues](https://github.com/GPflow/GPflow/issues/) to flag up bugs/issues/pain points, suggest new features, and discuss anything else related to the use of GPflow that in some sense involves changing the GPflow code itself.
You can make use of the [labels](https://github.com/GPflow/GPflow/labels) such as [`bug`](https://github.com/GPflow/GPflow/labels/bug), [`discussion`](https://github.com/GPflow/GPflow/labels/discussion), [`feature`](https://github.com/GPflow/GPflow/labels/feature), [`feedback`](https://github.com/GPflow/GPflow/labels/feedback), etc.
We positively welcome comments or concerns about usability, and suggestions for changes at any level of design.

We aim to respond to issues promptly, but if you believe we may have forgotten about an issue, please feel free to add another comment to remind us.

**"How-to-use" questions:**
Please use [Stack Overflow (gpflow tag)](https://stackoverflow.com/tags/gpflow) to ask questions that relate to "how to use GPflow", i.e. questions of understanding rather than issues that require changing GPflow code. (If you are unsure where to ask, you are always welcome to open a GitHub issue; we may then ask you to move your question to Stack Overflow.)

### Slack workspace

We have a public [GPflow slack workspace](https://gpflow.slack.com/). Please use this [invite link](https://join.slack.com/t/gpflow/shared_invite/enQtOTE5MDA0Nzg5NjA2LTYwZWI3MzhjYjNlZWI1MWExYzZjMGNhOWIwZWMzMGY0YjVkYzAyYjQ4NjgzNDUyZTgyNzcwYjAyY2QzMWRmYjE) if you'd like to join, whether to ask short informal questions or to be involved in the discussion and future development of GPflow.

### Contributing

All constructive input is gratefully received. For more information, see the [notes for contributors](contributing.md).

### Projects using GPflow

A few projects building on GPflow and demonstrating its usage are listed below.

| Project | Description |
| --- | --- |
| [GPflowOpt](https://github.com/GPflow/GPflowOpt)       | Bayesian Optimization using GPflow. |
| [VFF](https://github.com/jameshensman/VFF)       | Variational Fourier Features for Gaussian Processes. |
| [Doubly-Stochastic-DGP](https://github.com/ICL-SML/Doubly-Stochastic-DGP)| Deep Gaussian Processes with Doubly Stochastic Variational Inference.|
| [BranchedGP](https://github.com/ManchesterBioinference/BranchedGP) | Gaussian processes with branching kernels.|
| [heterogp](https://github.com/Joshuaalbert/heterogp) | Heteroscedastic noise for sparse variational GP. |
| [widedeepnetworks](https://github.com/widedeepnetworks/widedeepnetworks) | Measuring the relationship between random wide deep neural networks and GPs.| 
| [orth_decoupled_var_gps](https://github.com/hughsalimbeni/orth_decoupled_var_gps) | Variationally sparse GPs with orthogonally decoupled bases| 
| [kernel_learning](https://github.com/frgsimpson/kernel_learning) | Implementation of "Differentiable Compositional Kernel Learning for Gaussian Processes".|
| [VBPP](https://github.com/st--/vbpp) | Implementation of "Variational Bayes for Point Processes".|
| [DGPs_with_IWVI](https://github.com/hughsalimbeni/DGPs_with_IWVI) | Deep Gaussian Processes with Importance-Weighted Variational Inference|
| [kerndisc](https://github.com/BracketJohn/kernDisc) | Library for automated kernel structure discovery in univariate data|

If you would like your project listed here, let us know - or simply [open a pull request](https://github.com/GPflow/GPflow/compare) that adds your project to the table above!

## Compatibility

GPflow heavily depends on TensorFlow and as far as TensorFlow supports forward compatibility, GPflow should as well. The version of GPflow can give you a hint about backward compatibility. If the major version has changed then you need to check the release notes to find out how the API has been changed.

Unfortunately, there is no such thing as backward compatibility for GPflow _models_, which means that a model implementation can change without changing interfaces. In other words, the TensorFlow graph can be different for the same models from different versions of GPflow.

## Citing GPflow

To cite GPflow, please reference the [JMLR paper](http://www.jmlr.org/papers/volume18/16-537/16-537.pdf). Sample Bibtex is given below:

```
@ARTICLE{GPflow2017,
  author = {Matthews, Alexander G. de G. and {van der Wilk}, Mark and Nickson, Tom and
	Fujii, Keisuke. and {Boukouvalas}, Alexis and {Le{\'o}n-Villagr{\'a}}, Pablo and
	Ghahramani, Zoubin and Hensman, James},
    title = "{{GP}flow: A {G}aussian process library using {T}ensor{F}low}",
  journal = {Journal of Machine Learning Research},
  year    = {2017},
  month = {apr},
  volume  = {18},
  number  = {40},
  pages   = {1-6},
  url     = {http://jmlr.org/papers/v18/16-537.html}
}
```

Since the publication of the GPflow paper, the software has been significantly extended
with the framework for interdomain approximations and multioutput priors. We review the
framework and describe the design in an [arXiv paper](https://arxiv.org/abs/2003.01115),
which can be cited by users.

```
@article{GPflow2020multioutput,
  author = {{van der Wilk}, Mark and Dutordoir, Vincent and John, ST and
            Artemev, Artem and Adam, Vincent and Hensman, James},
  title = {A Framework for Interdomain and Multioutput {G}aussian Processes},
  year = {2020},
  journal = {arXiv:2003.01115},
  url = {https://arxiv.org/abs/2003.01115}
}
```
