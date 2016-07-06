# GPflow

GPflow is a package for building Gaussian process models in python, using [TensorFlow](http://www.tensorflow.org). It was originally created and is now managed by [James Hensman](http://www.lancaster.ac.uk/staff/hensmanj/) and [Alexander G. de G. Matthews](http://mlg.eng.cam.ac.uk/?portfolio=alex-matthews). 
The full list of [contributors](http://github.com/GPflow/GPflow/graphs/contributors) (in alphabetical order) is Alexis Boukouvalas, Keisuke Fujii, James Hensman, Pablo Leon, Alexander G. de G. Matthews, Valentine Svensson and Mark van der Wilk. GPflow is an open source project so if you feel you have some relevant skills and are interested in contributing then please do contact us.  

[![Build status](https://codeship.com/projects/26b43920-e96e-0133-3481-02cde9680eda/status?branch=master)](https://codeship.com/projects/147609)
[![Coverage Status](https://coveralls.io/repos/github/GPflow/GPflow/badge.svg?branch=HEAD)](https://coveralls.io/github/GPflow/GPflow?branch=HEAD)

# Install

## 1) Install TensorFlow. 
Please see instructions on the main TensorFlow [webpage](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#download-and-setup). You will need at least version 0.9 . We find that for many users pip installation is the fastest way to get going.

## 2) install package
GPflow is a pure python library for now, so you could just add it to your path (we use `python setup.py develop`) or try an install `python setup.py install` (untested). You can run the tests with `python setup.py test`.

# What's the difference between GPy and GPflow?

GPflow has origins in [GPy](http://github.com/sheffieldml/gpy) by the [GPy contributors](https://github.com/SheffieldML/GPy/graphs/contributors), and much of the interface is intentionally similar for continuity (though some parts of the interface may diverge in future). GPflow has a rather different remit from GPy though:

 -  GPflow attempts to leverage tensorflow for faster/bigger computation
 -  GPflow has much less code than GPy, mostly because all gradient computation is handled by tensorflow.
 -  GPflow focusses on variational inference and MCMC  -- there is no expectation propagation or Laplace approximation.
 -  GPflow does not do latent variable models (GPLVMs).
 -  GPflow does not have any plotting functionality.
 -  GPflow is not meant as a tool to teach about GPs. [GPy is much better at that](http://gpss.cc). 

# What models are implemented?
GPflow has a slew of kernels that can be combined in a similar way to GPy ([see this tutorial](https://github.com/SheffieldML/notebook/blob/master/GPy/basic_kernels.ipynb)). As for inference, the options are currently:

#### Regression
For GP regression with Gaussian noise, it's possible to marginalize the function values exactly: you'll find this in `GPflow.gpr.GPR`. You can do maximum likelihood or MCMC for the covariance function parameters ([notebook](https://github.com/GPflow/GPflow/blob/master/notebooks/regression.ipynb)).

It's also possible to do Sparse GP regression using the `GPflow.sgpr.SGPR` class. This is based on [4].

#### MCMC
For non-Gaussian likelihoods, GPflow has a model that can jointly sample over the function values and the covariance parameters: `GPflow.gpmc.GPMC`. There's also a sparse equivalent in `GPflow.sgpmc.SGPMC`, based on a recent paper [1]. This [notebook](https://github.com/GPflow/GPflow/blob/master/notebooks/Sparse%20mcmc%20demo.ipynb) introduces the interface.

#### Variational inference
It's often sufficient to approximate the function values as a Gaussian, for which we follow [2] in `GPflow.vgp.VGP`. In addition, there is a sparse version based on [3] in `GPflow.svgp.SVGP`. In the Gaussian likelihood case some of the optimization may be done analytically as discussed in [4] and implemented in `GPflow.sgpr.SGPR` . All of the sparse methods in GPflow are solidified in [5].

The following table summarizes the model options in GPflow. 

| | Gaussian <br> likelihood | Non-Gaussian <br> (variational) | Non-Gaussian <br> (MCMC)|
| --- | --- | --- | --- |
| Full-covariance | `GPflow.gpr.GPR` | `GPflow.vgp.VGP` | `GPflow.gpmc.GPMC`|
| Sparse approximation | `GPflow.sgpr.SGPR` | `GPflow.svgp.SVGP` | `GPflow.sgpmc.SGPMC` |

### References
[1] MCMC for Variationally Sparse Gaussian Processes
J Hensman, A G de G Matthews, M Filippone, Z Ghahramani
Advances in Neural Information Processing Systems, 1639-1647

[2] The variational Gaussian approximation revisited
M Opper, C Archambeau
Neural computation 21 (3), 786-792

[3] Scalable Variational Gaussian Process Classification
J Hensman, A G de G Matthews, Z Ghahramani
Proceedings of AISTATS 18, 2015

[4] Variational Learning of Inducing Variables in Sparse Gaussian Processes. 
M Titsias
Proceedings of AISTATS 12, 2009

[5] On Sparse variational methods and the Kullback-Leibler divergence between stochastic processes
A G de G Matthews, J Hensman, R E Turner, Z Ghahramani
Proceedings of AISTATS 19, 2016

### Acknowledgements

James Hensman was supported by an MRC fellowship and Alexander G. de G. Matthews was supported by EPSRC grant EP/I036575/1.
