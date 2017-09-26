# GPflow

GPflow is a package for building Gaussian process models in python, using [TensorFlow](http://www.tensorflow.org). It was originally created and is now managed by [James Hensman](http://www.lancaster.ac.uk/staff/hensmanj/) and [Alexander G. de G. Matthews](http://mlg.eng.cam.ac.uk/?portfolio=alex-matthews).
The full list of [contributors](http://github.com/GPflow/GPflow/graphs/contributors) (in alphabetical order) is Rasmus Bonnevie, Alexis Boukouvalas, Ivo Couckuyt, Keisuke Fujii, Zoubin Ghahramani, David J. Harris, James Hensman, Pablo Leon-Villagra, Daniel Marthaler, Alexander G. de G. Matthews, Tom Nickson, Valentine Svensson and Mark van der Wilk. GPflow is an open source project so if you feel you have some relevant skills and are interested in contributing then please do contact us.  

[![Python2.7 status](https://codeship.com/projects/26b43920-e96e-0133-3481-02cde9680eda/status?branch=master)](https://codeship.com/projects/147609)
[![Python3.5 Status](https://travis-ci.org/GPflow/GPflow.svg?branch=master)](https://travis-ci.org/GPflow/GPflow)
[![Coverage Status](http://codecov.io/github/GPflow/GPflow/coverage.svg?branch=master)](http://codecov.io/github/GPflow/GPflow?branch=master)
[![Documentation Status](https://readthedocs.org/projects/gpflow/badge/?version=latest)](http://gpflow.readthedocs.io/en/latest/?badge=latest)

# Coming soon: GPflow 1.0
We're working to improve GPflow, and @awav has undertaken a considerable reconfigurration of the code. There are several objectives:
 - make GPflow variable names line up with tensorflow scopes
 - line up GPflow params and tensorflow variables
 - allow GPflow to integrate into other projects (you can pass your own tensors into GPflow models)
 - better handling of tf.graphs and sessions
 - cleaner autoflow implementation
 - more transparent code
 
 This update will break backward-compatibility, but the current version of GPflow will still be available. Brave GPflowers can check out the code here: https://github.com/GPflow/GPflow/tree/GPflow-1.0-RC

# What does GPflow do?

GPflow implements modern Gaussian process inference for composable kernels and likelihoods. The [online user manual](http://gpflow.readthedocs.io/en/latest/) contains more details. The interface follows on from [GPy](http://github.com/sheffieldml/gpy), for more discussion of the comparison see [this page](http://gpflow.readthedocs.io/en/latest/intro.html#what-s-the-difference-between-gpy-and-gpflow).

# Install

## 1) Quick install
GPflow can be installed by cloning the repository and running
```
pip install .
```
in the root folder. This also installs required dependencies including TensorFlow. When GPU support is needed, a manual installation of TensorFlow is recommended (next section), as one cannot rely on pip to get this running.

## 2) Alternative method
A different option to install GPflow requires installation of TensorFlow first. Please see instructions on the main TensorFlow [webpage](https://www.tensorflow.org/versions/r1.0/get_started/get_started). You will need at least version 1.0 (we aim to support the latest version). We find that for most users pip installation is the fastest way to get going. Then, for those interested in modifying the source of GPflow, we recommend  
```
python setup.py develop
```
but installation should work well too:
```
python setup.py install
```
You can run the tests with `python setup.py test`.

Version history is documented [here.](https://github.com/GPflow/GPflow/blob/master/RELEASE.md)

# Deprecation

Python package name `GPflow` is no longer supported, it has been changed to lower-case name `gpflow`. You can adapt your code to new renamed package by running these commands:

```bash
## All files will be backed-up with `.bak` suffix
sed -i '.bak_import' 's/^\(import *\) GPflow/\1 gpflow/g' ./project-path
sed -i '.bak_from' 's/^\(from *\) GPflow/\1 gpflow/g' ./project-path
sed -i '.bak_dot' 's/GPflow\(\.[a-zA-Z0-9]\)/gpflow\1/g' ./project-path
```

## Docker image

We also provide a [Docker image](https://hub.docker.com/r/gpflow/gpflow/) which can be run using

```
docker run -it -p 8888:8888 gpflow/gpflow
```

Code to generate the image can be found [here](Dockerfile)

# Getting help
Please use gihub issues to start discussion on the use of GPflow. Tagging enquiries `discussion` helps us distinguish them from bugs.

# Contributing
All constuctive input is gratefully received. For more information, see the [notes for contributors](contributing.md).

# Projects using GPflow

A few projects building on GPflow and demonstrating its usage are listed below.

| Project | Description |
| --- | --- |
| [GPflowOpt](https://github.com/GPflow/GPflowOpt)       | Bayesian Optimization using GPflow. |
| [VFF](https://github.com/jameshensman/VFF)       | Variational Fourier Features for Gaussian Processes. |
| [Doubly-Stochastic-DGP](https://github.com/ICL-SML/Doubly-Stochastic-DGP)| Deep Gaussian Processes with Doubly Stochastic Variational Inference.|
| [BranchedGP](https://github.com/ManchesterBioinference/BranchedGP) | Gaussian processes with branching kernels.|

Let us know if you would like your project listed here.

# Citing GPflow

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
