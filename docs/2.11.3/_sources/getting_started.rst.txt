Getting Started
===============

This section aims to give you the knowledge necessary to use GPflow on small-to-medium projects,
without necessarily going too much into the mathematical and technical details. We do not try to
teach the theory behind Gaussian Processes.

* For a brief introduction to the mathematics of Gaussian Processes we recommend
  `this article <http://www.inference.org.uk/mackay/gpB.pdf>`_.
* For a longer text on the theory of Gaussian Processes we recommend the book:
  `Gaussian Processes for Machine Learning <https://gaussianprocess.org/gpml/>`_.
* If you need a deeper understanding of the technical details or advanced features of GPflow please,
  see our :doc:`user_guide`.

We will assume you are reasonably familiar with `Python <https://www.python.org/>`_,
`NumPy <https://numpy.org/>`_ and maybe `TensorFlow <https://www.tensorflow.org/>`_, and it is good
if you have some previous experience with data processing or machine learning in Python. We do not
assume prior knowldege of Gaussian Processes.


.. toctree::
   :maxdepth: 1

   installation
   notebooks/getting_started/basic_usage
   notebooks/getting_started/kernels
   notebooks/getting_started/mean_functions
   notebooks/getting_started/parameters_and_their_optimisation
   notebooks/getting_started/large_data
   notebooks/getting_started/classification_and_other_data_distributions
   notebooks/getting_started/monitoring
   notebooks/getting_started/saving_and_loading
