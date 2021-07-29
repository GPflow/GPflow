import tensorflow as tf
from typing import Optional

from .. import posteriors
from ..kernels import Kernel
from ..likelihoods.heteroskedastic import HeteroskedasticGaussianLikelihood
from ..mean_functions import MeanFunction
from ..models.gpr import GPR_with_posterior
from ..models.training_mixins import RegressionData
from ..utilities import add_noise_cov


class het_GPR(GPR_with_posterior):
    """ While the vanilla GPR enforces a constant noise variance across the input space, here we allow the
    noise amplitude to vary linearly (and hence the noise variance to change quadratically) across the input space.
    """

    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        mean_function: Optional[MeanFunction] = None,
        noise_variance: float = 1.0,
    ):
        super().__init__(data, kernel, mean_function, noise_variance)
        self.likelihood = HeteroskedasticGaussianLikelihood(noise_variance)

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
            likelihood_variance=self.likelihood.variance,
            mean_function=self.mean_function,
            precompute_cache=precompute_cache,
        )

    def _add_noise_cov(self, K: tf.Tensor) -> tf.Tensor:
        """
        Returns K + diag(σ²), where σ² is the likelihood noise variance (vector),
        and I is the corresponding identity matrix.
        """
        return add_noise_cov(K, self.likelihood.variance)



