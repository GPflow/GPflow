
========================
The six core models of GPflow
========================

Models are typically the highest level component that are used in GPflow; usually they comprise of at least one `kernel <kernel_options.html>`_ and at least one `likelihood <likelihood_options.html>`_. They additionally sometimes contain a `mean function <mean_function_options.html>`_, and `priors <prior_options.html>`_ over model parameters.

The following table summarizes the six core model options in GPflow.

+----------------------+--------------------------+----------------------------+-----------------------------+
|                      | Gaussian                 | Non-Gaussian (variational) | Non-Gaussian                |
|                      | Likelihood               |                            | (MCMC)                      |
+======================+==========================+============================+=============================+
| Full covariance      | :class:`gpflow.models.GPR`  | :class:`gpflow.models.VGP`    | :class:`gpflow.models.GPMC`   |
+----------------------+--------------------------+----------------------------+-----------------------------+
| Sparse approximation | :class:`gpflow.models.SGPR`| :class:`gpflow.models.SVGP`  | :class:`gpflow.models.SGPMC` |
+----------------------+--------------------------+----------------------------+-----------------------------+

The GPLVM which adds latent variables is also included (`notebook <notebooks/GPLVM.html>`_).

GP Regression
-------------

.. automodule:: gpflow.models
.. autoclass:: gpflow.models.GPR

Sparse GP Regression
--------------------

See also the documentation of the `derivation  <notebooks/SGPR_notes.html>`_.

.. automodule:: gpflow.models
.. autoclass:: gpflow.models.SGPR

Variational Gaussian Approximation
----------------------------------

See also the documentation of the `derivation  <notebooks/VGP_notes.html>`_.

.. automodule:: gpflow.models
.. autoclass:: gpflow.models.VGP

Sparse Variational Gaussian Approximation
-----------------------------------------

.. automodule:: gpflow.models
.. autoclass:: gpflow.models.SVGP

Markov Chain Monte Carlo
------------------------

.. automodule:: gpflow.models
.. autoclass:: gpflow.models.GPMC

Sparse Markov Chain Monte Carlo
-------------------------------

.. automodule:: gpflow.models
.. autoclass:: gpflow.models.SGPMC
