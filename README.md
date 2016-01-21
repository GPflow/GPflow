# GPflow

GPflow is a package for building Gaussian process models in python, using [TensorFlow](github.com/tensorflow) by James Hensman and Alexander G. de G. Matthews. 


# Install

### 1) Install tensorflow fork.
To make Gaussian processes work, we've had to add some extra functionality to TensorFlow. You'll need to install our [fork](github.com/GPflow/tensorflow), by building the pip package. You will also need [Bazel](http://bazel.io/). The sequence of commands is:

```
pip remove tensorflow
git clone --recurse-submodules github.com/gpflow/tensorflow
cd tensorflow
bazel build -c opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install /tmp/tensorflow_pkg/<package name>
```

For more information see [this page](https://www.tensorflow.org/versions/master/get_started/os_setup.html#installing-from-sources).

### 2) install package
GPflow is a pure python library for now, so you could just add it to your path (we use `python setup.py develop`) or try an install `python setup.py install` (untested). 

# What's the difference between GPy and GPflow?

GPflow has origins in [GPy](http://github.com/sheffieldml/gpy) by the [GPy contributors](https://github.com/SheffieldML/GPy/graphs/contributors), and much of the interface is intentionally similar for continuity (though some parts of the interface may diverge in future). GPflow has a rather different remit from GPy though:

 -  GPflow attempts to leverage tensorflow for faster/bigger computation
 -  GPflow has much less code than GPy, mostly because all gradient computation is handled by tensorflow.
 -  GPflow focusses on variational inference and MCMC  -- there is no expectation propagation or Laplace approximation.
 -  GPflow does not do latent variable models (GPLVMs).
 -  GPflow does not have any plotting functionality.
 -  GPflow is not meant as a tool to teach about GPs. [GPy is much better at that](http://gpss.cc). 

# What models are implemented?
GPflow has a slew of kernels that cam be combined in a similar way to GPy ([see this tutorial](https://github.com/SheffieldML/notebook/blob/master/GPy/basic_kernels.ipynb)). As for inference, the options are currently:

#### Regression
For GP regression with Gaussian noise, it's possible to marginalize the function values exactly: you'll find this in `GPflow.gpr.GPR`. You can do maximum liklelihood or MCMC for the covariance function parameters ([notebook](https://github.com/GPflow/GPflow/blob/master/notebooks/regression.ipynb)).

#### MCMC
For non-Gaussian likelohoods, GPflow has a model that can jointly sample over the function values and the covariance parameters: `GPflow.gpmc.GPMC`. There's also a sparse equivalent in `GPflow.sgpmc.SGPMC`, based on a recent paper [1]. This [notebook](https://github.com/GPflow/GPflow/blob/master/notebooks/Sparse%20mcmc%20demo.ipynb) introduces the interface.

#### Variational inference
It's often sufficient to approximate the function values as a Gaussian, for which we follow [2] in `GPflow.vgp.VGP`. In addition, there is a sparse version based on [3] in `GPflow.svgp.SVGP`. All of the sparse methods in GPflow are solidified in [4]. 

[1] MCMC for Variationally Sparse Gaussian Processes
J Hensman, A G de G Matthews, M Filippone, Z Ghahramani
Advances in Neural Information Processing Systems, 1639-1647

[2] The variational Gaussian approximation revisited
M Opper, C Archambeau
Neural computation 21 (3), 786-792

[3] Scalable Variational Gaussian Process Classification
J Hensman, A G de G Matthews, Z Ghahramani
Proceedings of AISTATS 18, 2015

[4] On Sparse variational methods and the Kullback-Leibler divergence between stochastic processes
A G de G Matthews, J Hensman, R E Turner, Z Ghahramani
Proceedings of AISTATS 19, 2016
