------------
Introduction
------------

GPflow is a package for building Gaussian process models in python, using `TensorFlow <http://www.tensorflow.org>`_. It was originally created and is now managed by `James Hensman <http://www.lancaster.ac.uk/staff/hensmanj/>`_ and `Alexander G. de G. Matthews <http://mlg.eng.cam.ac.uk/?portfolio=alex-matthews>`_.
The full list of [contributors](http://github.com/GPflow/GPflow/graphs/contributors) (in alphabetical order) is Alexander G. de G. Matthews, Alexis Boukouvalas, Artem Artemev, Daniel Marthaler, David J. Harris, Hugh Salimbeni, Ivo Couckuyt, James Hensman, Keisuke Fujii, Mark van der Wilk, Mikhail Beck, Pablo Leon-Villagra, Rasmus Bonnevie, ST John, Tom Nickson, Valentine Svensson, Vincent Dutordoir, Zoubin Ghahramani. GPflow is an open source project so if you feel you have some relevant skills and are interested in contributing then please do contact us.

Install
-------

GPflow can be installed by cloning the repository and running ``pip install .`` in the root folder. This also installs required dependencies including TensorFlow, and sets everything up.

A different installation approach requires installation of TensorFlow first. Please see instructions on the main TensorFlow `webpage <https://www.tensorflow.org/versions/r1.0/get_started/get_started>`_. You will need version 1.0 or higher. We find that for many users pip installation is the fastest way to get going.
As GPflow is a pure python library for now, you could just add it to your path (we use ``python setup.py develop``) or try an install ``python setup.py install`` (untested). You can run the tests with ``python setup.py test``.

Version history is documented `here <https://github.com/GPflow/GPflow/blob/master/RELEASE.md>`_.


Getting Started
---------------
Get started with our `examples and tutorials <notebooks/intro.html>`_.


What's the difference between GPy and GPflow?
---------------------------------------------

GPflow has origins in `GPy <http://github.com/sheffieldml/gpy>`_ by the `GPy contributors <https://github.com/SheffieldML/GPy/graphs/contributors>`_, and much of the interface is intentionally similar for continuity (though some parts of the interface may diverge in future). GPflow has a rather different remit from GPy though:

 -  GPflow leverages TensorFlow for faster/bigger computation
 -  GPflow has much less code than GPy, mostly because all gradient computation is handled by TensorFlow.
 -  GPflow focusses on variational inference and MCMC  -- there is no expectation propagation or Laplace approximation.
 -  GPflow does not have any plotting functionality.

.. _implemented_models:

What models are implemented?
----------------------------
GPflow has a slew of kernels that can be combined in a straightforward way. See the later section on `Using kernels in GPflow`. As for inference, the options are currently:

Regression
""""""""""
For GP regression with Gaussian noise, it's possible to marginalize the function values exactly: you'll find this in `gpflow.models.GPR`. You can do maximum likelihood or MCMC for the covariance function parameters  (`notebook <notebooks/basics/regression.html>`_).

It's also possible to do Sparse GP regression using the :class:`gpflow.models.SGPR` class. This is based on work by `Michalis Titsias <http://www.jmlr.org/proceedings/papers/v5/titsias09a.html>`_ [4].

MCMC
""""
For non-Gaussian likelihoods, GPflow has a model that can jointly sample over the function values and the covariance parameters: :class:`gpflow.models.GPMC`. There's also a sparse equivalent in :class:`gpflow.models.SGPMC`, based on a `recent paper <https://papers.nips.cc/paper/5875-mcmc-for-variationally-sparse-gaussian-processes>`_ [1].

Variational inference
"""""""""""""""""""""
It's often sufficient to approximate the function values as a Gaussian, for which we follow [2] in :class:`gpflow.models.VGP`. In addition, there is a sparse version based on [3] in :class:`gpflow.models.SVGP`. In the Gaussian likelihood case some of the optimization may be done analytically as discussed in [4] and implemented in :class:`gpflow.models.SGPR` . All of the sparse methods in GPflow are solidified in [5].

The following table summarizes the model options in GPflow.

+----------------------+----------------------------+----------------------------+------------------------------+
|                      | Gaussian                   | Non-Gaussian (variational) | Non-Gaussian                 |
|                      | Likelihood                 |                            | (MCMC)                       |
+======================+============================+============================+==============================+
| Full-covariance      | :class:`gpflow.models.GPR` | :class:`gpflow.models.VGP` | :class:`gpflow.models.GPMC`  |
+----------------------+----------------------------+----------------------------+------------------------------+
| Sparse approximation | :class:`gpflow.models.SGPR`| :class:`gpflow.models.SVGP`| :class:`gpflow.models.SGPMC` |
+----------------------+----------------------------+----------------------------+------------------------------+

A unified view of many of the relevant references, along with some extensions, and an early discussion of GPflow itself, is given in the PhD thesis of `Matthews <http://mlg.eng.cam.ac.uk/matthews/thesis.pdf>`_ [8].

Interdomain inference and multioutput GPs
"""""""""""""""""""""""""""""""""""""""""
GPflow has an extensive and flexible framework for specifying interdomain inducing variables for variational approximations.
Interdomain variables can greatly improve the effectiveness of a variational approximation, and are used in e.g.
`convolutional GPs <notebooks/advanced/convolutional.html>`_. In particular, they are crucial for defining sensible sparse
approximations for `multioutput GPs <notebooks/advanced/multioutput.html>`_.

GPflow has a unifying design for using multioutput GPs and specifying interdomain approximations. A review of the
mathematical background and the resulting software design is described in a `paper on arXiv <https://arxiv.org/abs/2003.01115>`_ [9].

GPLVM
"""""
For visualisation, the GPLVM [6] and Bayesian GPLVM [7] models are implemented
in GPflow (`notebook <notebooks/advanced/GPLVM.html>`_).

Contributing
------------
All constructive input is gratefully received. For more information, see the `notes for contributors <https://github.com/GPflow/GPflow/blob/master/contributing.md>`_.

Citing GPflow
-------------

To cite GPflow, please reference the `JMLR paper <http://www.jmlr.org/papers/volume18/16-537/16-537.pdf>`_. Sample BibTeX is given below:

.. code-block:: bib

    @ARTICLE{GPflow2017,
        author = {Matthews, Alexander G. de G. and {van der Wilk}, Mark and Nickson, Tom and Fujii, Keisuke. and {Boukouvalas}, Alexis and {Le{\'o}n-Villagr{\'a}}, Pablo and Ghahramani, Zoubin and Hensman, James},
        title = "{{GP}flow: A {G}aussian process library using {T}ensor{F}low}",
        journal = {Journal of Machine Learning Research},
        year    = {2017},
        month = {apr},
        volume  = {18},
        number  = {40},
        pages   = {1-6},
        url     = {http://jmlr.org/papers/v18/16-537.html}
    }

Since the publication of the GPflow paper, the software has been significantly extended
with the framework for interdomain approximations and multioutput priors. We review the
framework and describe the design in an `arXiv paper <https://arxiv.org/abs/2003.01115>`_
which can be cited by users.

.. code-block:: bib

    @article{GPflow2020multioutput,
      author = {{van der Wilk}, Mark and Dutordoir, Vincent and John, ST and
                Artemev, Artem and Adam, Vincent and Hensman, James},
      title = {A Framework for Interdomain and Multioutput {G}aussian Processes},
      year = {2020},
      journal = {arXiv:2003.01115},
      url = {https://arxiv.org/abs/2003.01115}
    }




References
----------
[1] MCMC for Variationally Sparse Gaussian Processes
J Hensman, A G de G Matthews, M Filippone, Z Ghahramani
Advances in Neural Information Processing Systems, 1639-1647, 2015.

[2] The variational Gaussian approximation revisited
M Opper, C Archambeau
Neural computation 21 (3), 786-792, 2009.

[3] Scalable Variational Gaussian Process Classification
J Hensman, A G de G Matthews, Z Ghahramani
Proceedings of AISTATS 18, 2015.

[4] Variational Learning of Inducing Variables in Sparse Gaussian Processes.
M Titsias
Proceedings of AISTATS 12, 2009.

[5] On Sparse variational methods and the Kullback-Leibler divergence between stochastic processes
A G de G Matthews, J Hensman, R E Turner, Z Ghahramani
Proceedings of AISTATS 19, 2016.

[6] Gaussian process latent variable models for visualisation of high dimensional data.
Lawrence, Neil D.
Advances in Neural Information Processing Systems, 329-336, 2004.

[7] Bayesian Gaussian Process Latent Variable Model.
Titsias, Michalis K., and Neil D. Lawrence.
Proceedings of AISTATS, 2010.

[8] Scalable Gaussian process inference using variational methods.
Alexander G. de G. Matthews.
PhD Thesis. University of Cambridge, 2016.

[9] A Framework for Interdomain and Multioutput Gaussian Processes.
Mark van der Wilk, Vincent Dutordoir, ST John, Artem Artemev, Vincent Adam, James Hensman.
arXiv. 2020.


Acknowledgements
----------------

James Hensman was supported by an MRC fellowship and Alexander G. de G. Matthews was supported by EPSRC grants EP/I036575/1 and EP/N014162/1.
