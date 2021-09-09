import tensorflow as tf
from typing import Optional

from gpflow import posteriors
from ..kernels import Kernel
from ..logdensities import multivariate_normal
from ..mean_functions import MeanFunction
from ..models.gpr import GPR_with_posterior
from ..models.training_mixins import RegressionData, InputData
from ..types import MeanAndVariance
from ..utilities import add_linear_noise_cov


class het_GPR(GPR_with_posterior):
    """ While the vanilla GPR enforces a constant noise variance across the input space, here we allow the
    noise amplitude to vary linearly (and hence the noise variance to change quadratically) across the input space.
    """

    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        likelihood,
        mean_function: Optional[MeanFunction] = None,
        noise_variance: float = 1.0,
    ):
        super().__init__(data, kernel, mean_function, noise_variance)
        self.likelihood = likelihood

    def posterior(self, precompute_cache=posteriors.PrecomputeCacheType.TENSOR):
        """
        Create the Posterior object which contains precomputed matrices for
        faster prediction.

        precompute_cache has three settings:

        - `PrecomputeCacheType.TENSOR` (or `"tensor"`): Precomputes the cached
          quantities and stores them as tensors (which allows differentiating
          through the prediction). This is the default.
        - `PrecomputeCacheType.VARIABLE` (or `"variable"`): Precomputes the cached
          quantities and stores them as variables, which allows for updating
          their values without changing the compute graph (relevant for AOT
          compilation).
        - `PrecomputeCacheType.NOCACHE` (or `"nocache"` or `None`): Avoids
          immediate cache computation. This is useful for avoiding extraneous
          computations when you only want to call the posterior's
          `fused_predict_f` method.
        """

        X, Y = self.data

        return posteriors.HeteroskedasticGPRPosterior(
            kernel=self.kernel,
            X_data=X,
            Y_data=Y,
            likelihood=self.likelihood,
            mean_function=self.mean_function,
            precompute_cache=precompute_cache,
        )

    def predict_y(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Compute the mean and variance of the held-out data at the input points.
        """
        if full_cov or full_output_cov:
            # See https://github.com/GPflow/GPflow/issues/1461
            raise NotImplementedError(
                "The predict_y method currently supports only the argument values full_cov=False and full_output_cov=False"
            )

        f_mean, f_var = self.predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        Fs = tf.concat([f_mean, Xnew], axis=-1)
        dummy_f_var = tf.zeros_like(f_var)
        F_vars = tf.concat([f_var, dummy_f_var], axis=-1)
        return self.likelihood.predict_mean_and_var(Fs, F_vars)

    def _add_noise_cov(self, X, K: tf.Tensor) -> tf.Tensor:
        """
        Returns K + diag(σ²), where σ² is the likelihood noise variance (vector),
        and I is the corresponding identity matrix.
        """
        dummy_F = tf.zeros_like(X)
        Fs = tf.concat([dummy_F, X], axis=-1)
        variances = self.likelihood.conditional_variance(Fs)
        return add_linear_noise_cov(K, tf.squeeze(variances))

    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        """
        X, Y = self.data
        K = self.kernel(X)
        ks = self._add_noise_cov(X, K)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(X)

        # [R,] log-likelihoods for each independent dimension of Y
        log_prob = multivariate_normal(Y, m, L)
        return tf.reduce_sum(log_prob)



