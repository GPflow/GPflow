---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# GPflow manual

<!-- #region -->
You can use this document to get familiar with GPflow. We've split up the material into four different categories: basics, understanding, advanced needs, and tailored models. We have also provided a [flow diagram](GPflows.png) to guide you to the relevant parts of GPflow for your specific problem.

## GPflow 2

Users of GPflow 1 should check the [upgrade guide to GPflow 2](gpflow2_upgrade_guide.ipynb).

## Basics

This section covers the elementary uses of GPflow, and shows you how to use GPflow for your basic datasets with existing models.

  - In [regression.ipynb](basics/regression.ipynb) and [classification.ipynb](basics/classification.ipynb) we show how to use GPflow to fit simple regression and classification models (Rasmussen and Williams, 2006).
  - In [gplvm.ipynb](basics/GPLVM.ipynb) we cover the unsupervised case, and showcase GPflow's Bayesian Gaussian Process Latent Variable Model (GPLVM) (Titsias and Lawrence, 2010).

In each notebook we go over the data format, model setup, model optimization, and prediction options.

## Understanding

This section covers the building blocks of GPflow from an implementation perspective, and shows how the different modules interact as a whole.

<!--  - [Architecture](understanding/architecture.ipynb)  **[TODO]** -->
<!--  - [Utilities](understanding/utilities.ipynb): expectations, multi-output, conditionals, Kullback-Leibler divergences (KL), log-densities, features and quadrature  **[TODO]** -->
  - [Manipulating models](understanding/models.ipynb)
  - [GPflow with TensorFlow 2](intro_to_gpflow2.ipynb) for handling datasets, training, monitoring, and checkpointing.


## Advanced needs

This section explains the more complex models and features that are available in GPflow.

### Models

  - [Markov Chain Monte Carlo (MCMC)](advanced/mcmc.ipynb): using Hamiltonian Monte Carlo to sample the posterior GP and hyperparameters.
  - [Ordinal regression](advanced/ordinal_regression.ipynb): using GPflow to deal with ordinal variables.
  - [Gaussian process regression with varying output noise](advanced/varying_noise.ipynb) for different data points, using a custom likelihood or the `SwitchedLikelihood`, and [Heteroskedastic regression with a multi-latent likelihood](advanced/heteroskedastic.ipynb).
  - [Multiclass classification](advanced/multiclass_classification.ipynb) for non-binary examples.
  - [GPs for big data](advanced/gps_for_big_data.ipynb): using GPflow's Sparse Variational Gaussian Process (SVGP) model (Hensman et al., 2013; 2015). Use sparse methods when dealing with large datasets (more than around a thousand data points).
<!--  - [GPs for big data (part 2)](advanced/advanced_many_points.ipynb)  **[TODO]** -->
  - [Multi-output models with coregionalisation](advanced/coregionalisation.ipynb): for when not all outputs are observed at every data point.
  - [Multi-output models with SVGPs](advanced/multioutput.ipynb): more efficient when all outputs are observed at all data points.
  - [Inter-domain Variational Fourier features](advanced/variational_fourier_features.ipynb): how to add new inter-domain inducing variables, at the example of representing sparse GPs in the spectral domain.
  - [Manipulating kernels](advanced/kernels.ipynb): information on the covariances that are included in the library, and how you can combine them to create new ones.
  - [Convolutional GPs](advanced/convolutional.ipynb): how we can use GPs with convolutional kernels for image classification.

### Features

  - [Natural gradients](advanced/natural_gradients.ipynb): how to optimize the variational approximate posterior's parameters.
  - [Monitoring optimisation](basics/monitoring.ipynb): how to monitor the model during optimisation: running custom callbacks and writing images and model parameters to TensorBoards.
<!--  - [optimizers](advanced/optimisation.ipynb)  **[TODO]** -->
<!--  - [Settings and GPflow configuration](advanced/settings.ipynb): how to adjust jitter (for inversion or Cholesky errors), floating point precision, parallelism, and more.-->

## Tailored models

This section shows how to use GPflow's utilities and codebase to build new probabilistic models.
These can be seen as complete examples.

  - [Kernel design](tailor/kernel_design.ipynb): how to implement a covariance function that is not available by default in GPflow. For this example, we look at the Brownian motion covariance.
<!--  - [likelihood design](tailor/likelihood_design.ipynb) **[TODO]** -->
<!--  - [Latent variable models](tailor/models_with_latent_variables.ipynb) **[TODO]** -->
<!--  - [Updating models with new data](tailor/updating_models_with_new_data.ipynb) **[TODO]** -->
  - [Mixing TensorFlow models with GPflow](tailor/gp_nn.ipynb): two ways to combine TensorFlow neural networks with GPflow models.
  - [External mean functions](tailor/external-mean-function.ipynb): how to use a neural network as a mean function.
  - [Mixture density network](tailor/mixture_density_network.ipynb): how GPflow's utilities make it easy to build other, non-GP probabilistic models.


## Theoretical notes

The following notebooks relate to the theory of Gaussian processes and approximations. These are not required reading for using GPflow, but are included for those interested in the theoretical underpinning and technical details.

  - [Derivation of VGP equations](theory/vgp_notes.ipynb)
  - [Derivation of SGPR equations](theory/SGPR_notes.ipynb)
  - [Demonstration of the upper bound of the SGPR marginal likelihood](theory/upper_bound.ipynb)
  - [Comparing FITC approximation to VFE approximation](theory/FITCvsVFE.ipynb): why we like the Variational Free Energy (VFE) objective rather than the Fully Independent Training Conditional (FITC) approximation for our sparse approximations.
  - A ['Sanity check' notebook](theory/Sanity_check.ipynb) that demonstrates the overlapping behaviour of many of the GPflow model classes in special cases (specifically, with a Gaussian likelihood and, for sparse approximations, inducing points fixed to the data points).

## References
Carl E Rasmussen and Christopher KI Williams. *Gaussian Processes for Machine Learning*. MIT Press, 2006.

James Hensman, Nicolo Fusi, and Neil D Lawrence. 'Gaussian Processes for Big Data'. *Uncertainty in Artificial Intelligence*, 2013.

James Hensman, Alexander G de G Matthews, and Zoubin Ghahramani. 'Scalable variational Gaussian process classification'. *Proceedings of the Eighteenth International Conference on Artificial Intelligence and Statistics*, 2015.

Michalis Titsias and Neil D Lawrence. 'Bayesian Gaussian process latent variable model'. *Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics*, 2010.
