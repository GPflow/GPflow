
========================
The six core models of GPflow
========================

The following table summarizes the six core model options in GPflow. 

+----------------------+--------------------------+----------------------------+-----------------------------+
|                      | Gaussian                 | Non-Gaussian (variational) | Non-Gaussian                |
|                      | Likelihood               |                            | (MCMC)                      |
+======================+==========================+============================+=============================+
| Full-covariance      | :class:`gpflow.gpr.GPR`  | :class:`gpflow.vgp.VGP`    | :class:`gpflow.gpmc.GPMC`   |
+----------------------+--------------------------+----------------------------+-----------------------------+
| Sparse approximation | :class:`gpflow.sgpr.SGPR`| :class:`gpflow.svgp.SVGP`  | :class:`gpflow.sgpmc.SGPMC` |
+----------------------+--------------------------+----------------------------+-----------------------------+

The GPLVM, which add latent variables is also included (`notebook <notebooks/GPLVM.html>`_).

GP Regression
-------------

.. automodule:: gpflow.gpr
.. autoclass:: gpflow.gpr.GPR

Sparse GP Regression
--------------------

See also the documentation of the `derivation  <notebooks/SGPR_notes.html>`_.

.. automodule:: gpflow.sgpr
.. autoclass:: gpflow.sgpr.SGPR

Variational Gaussian Approximation
----------------------------------

See also the documentation of the `derivation  <notebooks/VGP_notes.html>`_.

.. automodule:: gpflow.vgp
.. autoclass:: gpflow.vgp.VGP

Sparse Variational Gaussian Approximation
-----------------------------------------

.. automodule:: gpflow.svgp
.. autoclass:: gpflow.svgp.SVGP

Markov Chain Monte Carlo
------------------------

.. automodule:: gpflow.gpmc
.. autoclass:: gpflow.gpmc.GPMC

Sparse Markov Chain Monte Carlo
-------------------------------

.. automodule:: gpflow.sgpmc
.. autoclass:: gpflow.sgpmc.SGPMC
