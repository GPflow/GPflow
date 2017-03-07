
========================
The six core models of GPflow
========================

The following table summarizes the six core model options in GPflow. 

+----------------------+--------------------------+----------------------------+-----------------------------+
|                      | Gaussian                 | Non-Gaussian (variational) | Non-Gaussian                |
|                      | Likelihood               |                            | (MCMC)                      |
+======================+==========================+============================+=============================+
| Full-covariance      | :class:`GPflow.gpr.GPR`  | :class:`GPflow.vgp.VGP`    | :class:`GPflow.gpmc.GPMC`   |
+----------------------+--------------------------+----------------------------+-----------------------------+
| Sparse approximation | :class:`GPflow.sgpr.SGPR`| :class:`GPflow.svgp.SVGP`  | :class:`GPflow.sgpmc.SGPMC` |
+----------------------+--------------------------+----------------------------+-----------------------------+

The GPLVM, which add latent variables is also included (`notebook <notebooks/GPLVM.html>`_).

GP Regression
-------------

.. automodule:: GPflow.gpr
.. autoclass:: GPflow.gpr.GPR

Sparse GP Regression
--------------------

See also the documentation of the `derivation  <notebooks/SGPR_notes.html>`_.

.. automodule:: GPflow.sgpr
.. autoclass:: GPflow.sgpr.SGPR

Variational Gaussian Approximation
----------------------------------

See also the documentation of the `derivation  <notebooks/VGP_notes.html>`_.

.. automodule:: GPflow.vgp
.. autoclass:: GPflow.vgp.VGP

Sparse Variational Gaussian Approximation
-----------------------------------------

.. automodule:: GPflow.svgp
.. autoclass:: GPflow.svgp.SVGP

Markov Chain Monte Carlo
------------------------

.. automodule:: GPflow.gpmc
.. autoclass:: GPflow.gpmc.GPMC

Sparse Markov Chain Monte Carlo
-------------------------------

.. automodule:: GPflow.sgpmc
.. autoclass:: GPflow.sgpmc.SGPMC
