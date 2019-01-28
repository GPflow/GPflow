==================
Likelihoods included in GPflow
==================

Likelihoods are another core component of GPflow. This describes how likely the data is under the assumptions made about the underlying latent functions :math:`p(\mathbf{Y}|\mathbf{F})`. Different likelihoods make different assumptions about the distribution of the data, as such different data-types (continuous, binary, ordinal, count) are better modelled with different likelihood assumptions.

Use of any likelihood other than Gaussian typically introduces the need to use an approximation to perform inference, if one isn't already needed. A variational inference and MCMC models are included in GPflow and allow approximate inference with non-Gaussian likelihoods. An introduction to these models can be found :ref:`here <implemented_models>`. Specific notebooks illustrating non-Gaussian likelihood regressions are available for `classification <notebooks/classification.html>`_ (binary data), `ordinal <notebooks/ordinal.html>`_ and `multiclass <notebooks/multiclass.html>`_.

Creating new likelihoods
----------
Likelihoods are defined by their log-likelihood. When creating new likelihoods, the :func:`logp <gpflow.likelihoods.Likelihood.logp>` method (:math:`\log p(\mathbf{Y}|\mathbf{F})`), the :func:`conditional_mean <gpflow.likelihoods.Likelihood.conditional_mean>`, :func:`conditional_variance <gpflow.likelihoods.Likelihood.conditional_variance>`.

In order to perform variational inference with non-Gaussian likelihoods a term called ``variational expectations``, :math:`\int q(\mathbf{F})\log p(\mathbf{Y}|\mathbf{F}) d\mathbf{F}`, needs to be computed under a Gaussian distribution :math:`q(\mathbf{F}) \sim N(\mathbf{\mu}, \mathbf{\Sigma})`. 

The :func:`variational_expectations <gpflow.likelihoods.Likelihood.variational_expectations>` method can be overriden if this can be computed in closed form, otherwise; if the new likelihood inherits :class:`Likelihood <gpflow.likelihoods.Likelihood>` the default will use Gauss-Hermite numerical integration (works well when :math:`\mathbf{F}` is 1D or 2D), if the new likelihood inherits from :class:`MonteCarloLikelihood <gpflow.likelihoods.MonteCarloLikelihood>` the integration (can be more suitable when :math:`\mathbf{F}` is higher dimensional).

Likelihoods
-------
.. automodule:: gpflow.likelihoods
    :members:
