
========================
The six core models of GPflow
========================

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
