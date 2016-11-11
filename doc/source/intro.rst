------------
Introduction
------------

GPflow is a package for building Gaussian process models in python, using `TensorFlow <http://www.tensorflow.org>`_. It was originally created and is now managed by `James Hensman <http://www.lancaster.ac.uk/staff/hensmanj/>`_ and `Alexander G. de G. Matthews <http://mlg.eng.cam.ac.uk/?portfolio=alex-matthews>`_. 
The full list of `contributors <http://github.com/GPflow/GPflow/graphs/contributors>`_ (in alphabetical order) is Alexis Boukouvalas, Ivo Couckuyt, Keisuke Fujii, Zoubin Ghahramani, David J. Harris, James Hensman, Pablo Leon-Villagra, Daniel Marthaler, Alexander G. de G. Matthews, Tom Nickson, Valentine Svensson and Mark van der Wilk. GPflow is an open source project so if you feel you have some relevant skills and are interested in contributing then please do contact us.  

Install
-------

1. Install TensorFlow. 
Please see instructions on the main TensorFlow `webpage <https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#download-and-setup>`_. You will need at least version 0.10 . We find that for many users pip installation is the fastest way to get going.

2. install package
GPflow is a pure python library for now, so you could just add it to your path (we use ``python setup.py develop``) or try an install ``python setup.py install`` (untested). You can run the tests with ``python setup.py test``.

What's the difference between GPy and GPflow?
---------------------------------------------

GPflow has origins in `GPy <http://github.com/sheffieldml/gpy>`_ by the `GPy contributors <https://github.com/SheffieldML/GPy/graphs/contributors>`_, and much of the interface is intentionally similar for continuity (though some parts of the interface may diverge in future). GPflow has a rather different remit from GPy though:

 -  GPflow leverages TensorFlow for faster/bigger computation
 -  GPflow has much less code than GPy, mostly because all gradient computation is handled by TensorFlow.
 -  GPflow focusses on variational inference and MCMC  -- there is no expectation propagation or Laplace approximation.
 -  GPflow does not have any plotting functionality.

What models are implemented?
----------------------------
GPflow has a slew of kernels that can be combined in a straightforward way. See the later section on `Using kernels in GPflow`. As for inference, the options are currently:

Regression
~~~~~~~~~~
For GP regression with Gaussian noise, it's possible to marginalize the function values exactly: you'll find this in `GPflow.gpr.GPR`. You can do maximum likelihood or MCMC for the covariance function parameters (`notebook <https://github.com/GPflow/GPflow/blob/master/notebooks/regression.ipynb>`_).

It's also possible to do Sparse GP regression using the :class:`GPflow.sgpr.SGPR` class. This is based on work by `Michalis Titsias <http://www.jmlr.org/proceedings/papers/v5/titsias09a.html>`_ [4].

MCMC
~~~~
For non-Gaussian likelihoods, GPflow has a model that can jointly sample over the function values and the covariance parameters: :class:`GPflow.gpmc.GPMC`. There's also a sparse equivalent in :class:`GPflow.sgpmc.SGPMC`, based on a `recent paper <https://papers.nips.cc/paper/5875-mcmc-for-variationally-sparse-gaussian-processes>`_ [1]. 

Variational inference
~~~~~~~~~~~~~~~~~~~~~
It's often sufficient to approximate the function values as a Gaussian, for which we follow [2] in :class:`GPflow.vgp.VGP`. In addition, there is a sparse version based on [3] in :class:`GPflow.svgp.SVGP`. In the Gaussian likelihood case some of the optimization may be done analytically as discussed in [4] and implemented in :class:`GPflow.sgpr.SGPR` . All of the sparse methods in GPflow are solidified in [5].

The following table summarizes the model options in GPflow. 

+----------------------+--------------------------+----------------------------+-----------------------------+
|                      | Gaussian                 | Non-Gaussian (variational) | Non-Gaussian                |
|                      | Likelihood               |                            | (MCMC)                      |
+======================+==========================+============================+=============================+
| Full-covariance      | :class:`GPflow.gpr.GPR`  | :class:`GPflow.vgp.VGP`    | :class:`GPflow.gpmc.GPMC`   |
+----------------------+--------------------------+----------------------------+-----------------------------+
| Sparse approximation | :class:`GPflow.sgpr.SGPR`| :class:`GPflow.svgp.SVGP`  | :class:`GPflow.sgpmc.SGPMC` |
+----------------------+--------------------------+----------------------------+-----------------------------+

# Citing GPflow

To cite GPflow, please reference the [Technical report](https://arxiv.org/abs/1610.08733). Sample Bibtex is given below:

```
@ARTICLE{GPflow2016,
   author = {Matthews, Alexander G. de G. and {van der Wilk}, Mark and Nickson, Tom and 
	Fujii, Keisuke. and {Boukouvalas}, Alexis and {Le{\'o}n-Villagr{\'a}}, Pablo and 
	Ghahramani, Zoubin and Hensman, James},
    title = "{{GP}flow: A {G}aussian process library using {T}ensor{F}low}",
  journal = {arXiv preprint 1610.08733},
     year = 2016,
    month = oct
}
```

References
----------
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

Acknowledgements
----------------

James Hensman was supported by an MRC fellowship and Alexander G. de G. Matthews was supported by EPSRC grants EP/I036575/1 and EP/N014162/1.
