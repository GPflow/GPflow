User Guide
==========

You can use this document to get familiar with the more in-depth topics of GPflow. If you are new to GPflow you should see our :doc:`getting_started` guide first. We have also provided a `flow diagram <_static/GPflows.png>`_ to guide you to the relevant parts of GPflow for your specific problem.

.. _implemented_models:

What models are implemented?
----------------------------
GPflow has a slew of kernels that can be combined in a straightforward way. As for inference, the options are currently:

Regression
""""""""""
For GP regression with Gaussian noise, it's possible to marginalize the function values exactly: you'll find this in :class:`gpflow.models.GPR`. You can do maximum likelihood or MCMC for the covariance function parameters.

It's also possible to do Sparse GP regression using the :class:`gpflow.models.SGPR` class. This is based on work by Michalis Titsias :cite:p:`titsias2009variational`.

MCMC
""""
For non-Gaussian likelihoods, GPflow has a model that can jointly sample over the function values and the covariance parameters: :class:`gpflow.models.GPMC`. There's also a sparse equivalent in :class:`gpflow.models.SGPMC`, based on :cite:t:`hensman2015mcmc`.

Variational inference
"""""""""""""""""""""
It's often sufficient to approximate the function values as a Gaussian, for which we follow :cite:t:`Opper:2009` in :class:`gpflow.models.VGP`. In addition, there is a sparse version based on :cite:t:`hensman2014scalable` in :class:`gpflow.models.SVGP`. In the Gaussian likelihood case some of the optimization may be done analytically as discussed in :cite:t:`titsias2009variational` and implemented in :class:`gpflow.models.SGPR` . All of the sparse methods in GPflow are solidified in :cite:t:`matthews2016sparse`.

The following table summarizes the model options in GPflow.

+----------------------+----------------------------+----------------------------+------------------------------+
|                      | Gaussian                   | Non-Gaussian (variational) | Non-Gaussian                 |
|                      | Likelihood                 |                            | (MCMC)                       |
+======================+============================+============================+==============================+
| Full-covariance      | :class:`gpflow.models.GPR` | :class:`gpflow.models.VGP` | :class:`gpflow.models.GPMC`  |
+----------------------+----------------------------+----------------------------+------------------------------+
| Sparse approximation | :class:`gpflow.models.SGPR`| :class:`gpflow.models.SVGP`| :class:`gpflow.models.SGPMC` |
+----------------------+----------------------------+----------------------------+------------------------------+

A unified view of many of the relevant references, along with some extensions, and an early discussion of GPflow itself, is given in the PhD thesis of Matthews :cite:p:`matthews2017scalable`.

Interdomain inference and multioutput GPs
"""""""""""""""""""""""""""""""""""""""""
GPflow has an extensive and flexible framework for specifying interdomain inducing variables for variational approximations.
Interdomain variables can greatly improve the effectiveness of a variational approximation, and are used in e.g.
:doc:`notebooks/advanced/convolutional`. In particular, they are crucial for defining sensible sparse
approximations for multioutput GPs (:doc:`notebooks/advanced/multioutput`).

GPflow has a unifying design for using multioutput GPs and specifying interdomain approximations. A review of the
mathematical background and the resulting software design is described in :cite:t:`GPflow2020multioutput`.

GPLVM
"""""
For visualisation, the GPLVM :cite:p:`lawrence2003gaussian` and Bayesian GPLVM :cite:p:`titsias2010bayesian` models are implemented
in GPflow (:doc:`notebooks/advanced/GPLVM`).



Theoretical notes
-----------------

The following notebooks relate to the theory of Gaussian processes and approximations. These are not required reading for using GPflow, but are included for those interested in the theoretical underpinning and technical details.

.. toctree::
   :maxdepth: 1

   notebooks/theory/vgp_notes
   notebooks/theory/SGPR_notes
   notebooks/theory/cglb
   notebooks/theory/upper_bound

.. toctree::
   :maxdepth: 1

   notebooks/theory/FITCvsVFE

Why we like the Variational Free Energy (VFE) objective rather than the Fully Independent Training Conditional (FITC) approximation for our sparse approximations.

.. toctree::
   :maxdepth: 1

   notebooks/theory/Sanity_check

Demonstrates the overlapping behaviour of many of the GPflow model classes in special cases (specifically, with a Gaussian likelihood and, for sparse approximations, inducing points fixed to the data points).


Tailored models
---------------

This section shows how to use GPflow's utilities and codebase to build new probabilistic models.
These can be seen as complete examples.

.. toctree::
   :maxdepth: 1

   notebooks/tailor/kernel_design

How to implement a covariance function that is not available by default in GPflow. For this example, we look at the Brownian motion covariance.

.. toctree::
   :maxdepth: 1

   notebooks/tailor/gp_nn

Two ways to combine TensorFlow neural networks with GPflow models.

.. toctree::
   :maxdepth: 1

   notebooks/tailor/external-mean-function

How to use a neural network as a mean function.

.. toctree::
   :maxdepth: 1

   notebooks/tailor/mixture_density_network

How GPflow's utilities make it easy to build other, non-GP probabilistic models.


Advanced needs
--------------

This section explains the more complex models and features that are available in GPflow.

.. toctree::
   :maxdepth: 1

   notebooks/advanced/mcmc

Using Hamiltonian Monte Carlo to sample the posterior GP and hyperparameters.

.. toctree::
   :maxdepth: 1

   notebooks/advanced/ordinal_regression

Using GPflow to deal with ordinal variables.

.. toctree::
   :maxdepth: 1

   notebooks/advanced/gps_for_big_data

Using GPflow's Sparse Variational Gaussian Process (SVGP) model :cite:p:`hensman2014scalable` :cite:p:`hensman2013gaussian`. Use sparse methods when dealing with large datasets (more than around a thousand data points).

.. toctree::
   :maxdepth: 1

   notebooks/advanced/multiclass_classification

On classification with more than two classes.

.. toctree::
   :maxdepth: 1

   notebooks/advanced/varying_noise

Most GP models assume the noise is constant across all data points. This notebook shows how to model simple varying noise.

.. toctree::
   :maxdepth: 1

   notebooks/advanced/heteroskedastic

This is a more expensive, but also more powerful way to handly noise that varies across data points.

.. toctree::
   :maxdepth: 1

   notebooks/advanced/changepoints

How to deal with regime-changes in 1D datasets.

.. toctree::
   :maxdepth: 1

   notebooks/advanced/convolutional

How we can use GPs with convolutional kernels for image classification.

.. toctree::
   :maxdepth: 1

   notebooks/advanced/multioutput

For when you have multiple outputs, that all are observed at all data points.

.. toctree::
   :maxdepth: 1

   notebooks/advanced/coregionalisation

For when you have multiple outputs, but not all of them are observed at every data point.

.. toctree::
   :maxdepth: 1

   notebooks/advanced/fast_predictions

How to use caching to speed up repeated predictions.

.. toctree::
   :maxdepth: 1

   notebooks/advanced/GPLVM

How to use the Bayesian GPLVM model. This is an unsupervised learning method usually used for dimensionality reduction.

.. toctree::
   :maxdepth: 1

   notebooks/advanced/variational_fourier_features

In this notebook we demonstrate how new types of inducing variables can easily be incorporated in the GPflow framework. As an example case, we use variational Fourier features.

.. toctree::
   :maxdepth: 1

   notebooks/advanced/natural_gradients

How to optimize the variational approximate posterior's parameters.
