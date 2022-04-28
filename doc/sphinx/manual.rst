-------------
GPflow manual
-------------

You can use this document to get familiar with GPflow. We've split up the material into four different categories: basics, understanding, advanced needs, and tailored models. We have also provided a `flow diagram <_static/GPflows.png>`_ to guide you to the relevant parts of GPflow for your specific problem.


Basics
------

This section covers the elementary uses of GPflow, and shows you how to use GPflow for your basic datasets with existing models.

- In :doc:`notebooks/basics/regression` and :doc:`notebooks/basics/classification` we show how to use GPflow to fit simple regression and classification models :cite:p:`rasmussen_williams_06`.
- In :doc:`notebooks/basics/GPLVM` we cover the unsupervised case, and showcase GPflow's Bayesian Gaussian Process Latent Variable Model (GPLVM) :cite:p:`titsias2010bayesian`.

In each notebook we go over the data format, model setup, model optimization, and prediction options.


Understanding
-------------

This section covers the building blocks of GPflow from an implementation perspective, and shows how the different modules interact as a whole.

- :doc:`notebooks/understanding/models`
- :doc:`notebooks/intro_to_gpflow2` for handling datasets, training, monitoring, and checkpointing.

.. **[TODO]** - :doc:`notebooks/understanding/architecture`
.. **[TODO]** - :doc:`notebooks/understanding/utilities`: expectations, multi-output, conditionals, Kullback-Leibler divergences (KL), log-densities, features and quadrature


Advanced needs
--------------

This section explains the more complex models and features that are available in GPflow.


Models
******

- :doc:`notebooks/advanced/mcmc`: using Hamiltonian Monte Carlo to sample the posterior GP and hyperparameters.
- :doc:`notebooks/advanced/ordinal_regression`: using GPflow to deal with ordinal variables.
- :doc:`notebooks/advanced/varying_noise` for different data points, using a custom likelihood or the `SwitchedLikelihood`, and :doc:`notebooks/advanced/heteroskedastic`.
- :doc:`notebooks/advanced/multiclass_classification` for non-binary examples.
- :doc:`notebooks/advanced/gps_for_big_data`: using GPflow's Sparse Variational Gaussian Process (SVGP) model :cite:p:`hensman2014scalable` :cite:p:`hensman2013gaussian`. Use sparse methods when dealing with large datasets (more than around a thousand data points).
- :doc:`notebooks/advanced/coregionalisation`: for when not all outputs are observed at every data point.
- :doc:`notebooks/advanced/multioutput`: more efficient when all outputs are observed at all data points.
- :doc:`notebooks/advanced/variational_fourier_features`: how to add new inter-domain inducing variables, at the example of representing sparse GPs in the spectral domain.
- :doc:`notebooks/advanced/kernels`: information on the covariances that are included in the library, and how you can combine them to create new ones.
- :doc:`notebooks/advanced/convolutional`: how we can use GPs with convolutional kernels for image classification.
- :doc:`notebooks/advanced/fast_predictions`: how to use caching to speed up repeated predictions.

.. **[TODO]** - :doc:`notebooks/advanced/advanced_many_points`


Features
********

- :doc:`notebooks/advanced/natural_gradients`: how to optimize the variational approximate posterior's parameters.
- :doc:`notebooks/basics/monitoring`: how to monitor the model during optimisation: running custom callbacks and writing images and model parameters to TensorBoards.

.. **[TODO]** - :doc:`notebooks/advanced/optimisation`
.. **[TODO]** - :doc:`notebooks/advanced/settings`: how to adjust jitter (for inversion or Cholesky errors), floating point precision, parallelism, and more.-->


Tailored models
---------------

This section shows how to use GPflow's utilities and codebase to build new probabilistic models.
These can be seen as complete examples.

- :doc:`notebooks/tailor/kernel_design`: how to implement a covariance function that is not available by default in GPflow. For this example, we look at the Brownian motion covariance.
- :doc:`notebooks/tailor/gp_nn`: two ways to combine TensorFlow neural networks with GPflow models.
- :doc:`notebooks/tailor/external-mean-function`: how to use a neural network as a mean function.
- :doc:`notebooks/tailor/mixture_density_network`: how GPflow's utilities make it easy to build other, non-GP probabilistic models.

.. **[TODO]** - :doc:`notebooks/tailor/likelihood_design`
.. **[TODO]** - :doc:`notebooks/tailor/models_with_latent_variables`
.. **[TODO]** - :doc:`notebooks/tailor/updating_models_with_new_data`



Theoretical notes
-----------------

The following notebooks relate to the theory of Gaussian processes and approximations. These are not required reading for using GPflow, but are included for those interested in the theoretical underpinning and technical details.

- :doc:`notebooks/theory/vgp_notes`
- :doc:`notebooks/theory/SGPR_notes`
- :doc:`notebooks/theory/upper_bound`
- :doc:`notebooks/theory/FITCvsVFE`: why we like the Variational Free Energy (VFE) objective rather than the Fully Independent Training Conditional (FITC) approximation for our sparse approximations.
- A :doc:`notebooks/theory/Sanity_check` that demonstrates the overlapping behaviour of many of the GPflow model classes in special cases (specifically, with a Gaussian likelihood and, for sparse approximations, inducing points fixed to the data points).
