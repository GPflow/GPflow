# Copyright 2016-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import tensorflow as tf

from ..base import MeanAndVariance, Module, TensorType
from ..experimental.check_shapes import check_shapes, inherit_check_shapes
from ..quadrature import GaussianQuadrature, NDiagGHQuadrature, ndiag_mc

DEFAULT_NUM_GAUSS_HERMITE_POINTS = 20
"""
The number of Gauss-Hermite points to use for quadrature (fallback when a
likelihood method does not have an analytic method) if quadrature object is not
explicitly passed to likelihood constructor.
"""


class Likelihood(Module, abc.ABC):
    def __init__(
        self, input_dim: Optional[int], latent_dim: Optional[int], observation_dim: Optional[int]
    ) -> None:
        """
        A base class for likelihoods, which specifies an observation model
        connecting the latent functions ('F') to the data ('Y').

        All of the members of this class are expected to obey some shape conventions, as specified
        by latent_dim and observation_dim.

        If we're operating on an array of function values 'F', then the last dimension represents
        multiple functions (preceding dimensions could represent different data points, or
        different random samples, for example). Similarly, the last dimension of Y represents a
        single data point. We check that the dimensions are as this object expects.

        The return shapes of all functions in this class is the broadcasted shape of the arguments,
        excluding the last dimension of each argument.

        :param input_dim: the dimension of the input vector X for a single data point
        :param latent_dim: the dimension of the vector F of latent functions for a single data point
        :param observation_dim: the dimension of the observation vector Y for a single data point
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.observation_dim = observation_dim

    @check_shapes(
        "X: [broadcast batch..., input_dim]",
        "F: [broadcast batch..., latent_dim]",
        "Y: [broadcast batch..., observation_dim]",
        "return: [batch...]",
    )
    def log_prob(self, X: TensorType, F: TensorType, Y: TensorType) -> tf.Tensor:
        """
        The log probability density log p(Y|X,F)

        :param X: input tensor
        :param F: function evaluation tensor
        :param Y: observation tensor
        :returns: log pdf
        """
        return self._log_prob(X, F, Y)

    @abc.abstractmethod
    @check_shapes(
        "X: [broadcast batch..., input_dim]",
        "F: [broadcast batch..., latent_dim]",
        "Y: [broadcast batch..., observation_dim]",
        "return: [batch...]",
    )
    def _log_prob(self, X: TensorType, F: TensorType, Y: TensorType) -> tf.Tensor:
        raise NotImplementedError

    @check_shapes(
        "X: [broadcast batch..., input_dim]",
        "F: [broadcast batch..., latent_dim]",
        "return: [batch..., observation_dim]",
    )
    def conditional_mean(self, X: TensorType, F: TensorType) -> tf.Tensor:
        """
        The conditional mean of Y|X,F: [E[Y₁|X,F], ..., E[Yₖ|X,F]]
        where K = observation_dim

        :param X: input tensor
        :param F: function evaluation tensor
        :returns: mean
        """
        return self._conditional_mean(X, F)

    @check_shapes(
        "X: [broadcast batch..., input_dim]",
        "F: [broadcast batch..., latent_dim]",
        "return: [batch..., observation_dim]",
    )
    def _conditional_mean(self, X: TensorType, F: TensorType) -> tf.Tensor:
        raise NotImplementedError

    @check_shapes(
        "X: [broadcast batch..., input_dim]",
        "F: [broadcast batch..., latent_dim]",
        "return: [batch..., observation_dim]",
    )
    def conditional_variance(self, X: TensorType, F: TensorType) -> tf.Tensor:
        """
        The conditional marginal variance of Y|X,F: [var(Y₁|X,F), ..., var(Yₖ|X,F)]
        where K = observation_dim

        :param X: input tensor
        :param F: function evaluation tensor
        :returns: variance
        """
        return self._conditional_variance(X, F)

    @check_shapes(
        "X: [broadcast batch..., input_dim]",
        "F: [broadcast batch..., latent_dim]",
        "return: [batch..., observation_dim]",
    )
    def _conditional_variance(self, X: TensorType, F: TensorType) -> tf.Tensor:
        raise NotImplementedError

    @check_shapes(
        "X: [broadcast batch..., input_dim]",
        "Fmu: [broadcast batch..., latent_dim]",
        "Fvar: [broadcast batch..., latent_dim]",
        "return[0]: [batch..., observation_dim]",
        "return[1]: [batch..., observation_dim]",
    )
    def predict_mean_and_var(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType
    ) -> MeanAndVariance:
        """
        Given a Normal distribution for the latent function,
        return the mean and marginal variance of Y,

        i.e. if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive mean

           ∫∫ y p(y|f)q(f) df dy

        and the predictive variance

           ∫∫ y² p(y|f)q(f) df dy  - [ ∫∫ y p(y|f)q(f) df dy ]²

        :param X: input tensor
        :param Fmu: mean function evaluation tensor
        :param Fvar: variance of function evaluation tensor
        :returns: mean and variance
        """
        return self._predict_mean_and_var(X, Fmu, Fvar)

    @abc.abstractmethod
    @check_shapes(
        "X: [broadcast batch..., input_dim]",
        "Fmu: [broadcast batch..., latent_dim]",
        "Fvar: [broadcast batch..., latent_dim]",
        "return[0]: [batch..., observation_dim]",
        "return[1]: [batch..., observation_dim]",
    )
    def _predict_mean_and_var(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType
    ) -> MeanAndVariance:
        raise NotImplementedError

    @check_shapes(
        "X: [broadcast batch..., input_dim]",
        "Fmu: [broadcast batch..., latent_dim]",
        "Fvar: [broadcast batch..., latent_dim]",
        "Y: [broadcast batch..., observation_dim]",
        "return: [batch...]",
    )
    def predict_log_density(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType
    ) -> tf.Tensor:
        r"""
        Given a Normal distribution for the latent function, and a datum Y,
        compute the log predictive density of Y,

        i.e. if
            q(F) = N(Fmu, Fvar)

        and this object represents

            p(y|F)

        then this method computes the predictive density

            log ∫ p(y=Y|F)q(F) df

        :param X: input tensor
        :param Fmu: mean function evaluation tensor
        :param Fvar: variance of function evaluation tensor
        :param Y: observation tensor
        :returns: log predictive density
        """
        return self._predict_log_density(X, Fmu, Fvar, Y)

    @abc.abstractmethod
    @check_shapes(
        "X: [broadcast batch..., input_dim]",
        "Fmu: [broadcast batch..., latent_dim]",
        "Fvar: [broadcast batch..., latent_dim]",
        "Y: [broadcast batch..., observation_dim]",
        "return: [batch...]",
    )
    def _predict_log_density(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType
    ) -> tf.Tensor:
        raise NotImplementedError

    @check_shapes(
        "X: [broadcast batch..., input_dim]",
        "Fmu: [broadcast batch..., latent_dim]",
        "Fvar: [broadcast batch..., latent_dim]",
        "Y: [broadcast batch..., observation_dim]",
        "return: [batch...]",
    )
    def variational_expectations(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType
    ) -> tf.Tensor:
        r"""
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values,

        i.e. if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes

           ∫ log(p(y=Y|f)) q(f) df.

        This only works if the broadcasting dimension of the statistics of q(f) (mean and variance)
        are broadcastable with that of the data Y.

        :param X: input tensor
        :param Fmu: mean function evaluation tensor
        :param Fvar: variance of function evaluation tensor
        :param Y: observation tensor
        :returns: expected log density of the data given q(F)
        """
        return self._variational_expectations(X, Fmu, Fvar, Y)

    @abc.abstractmethod
    @check_shapes(
        "X: [broadcast batch..., input_dim]",
        "Fmu: [broadcast batch..., latent_dim]",
        "Fvar: [broadcast batch..., latent_dim]",
        "Y: [broadcast batch..., observation_dim]",
        "return: [batch...]",
    )
    def _variational_expectations(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType
    ) -> tf.Tensor:
        raise NotImplementedError


class QuadratureLikelihood(Likelihood, abc.ABC):
    def __init__(
        self,
        input_dim: Optional[int],
        latent_dim: Optional[int],
        observation_dim: Optional[int],
        *,
        quadrature: Optional[GaussianQuadrature] = None,
    ) -> None:
        super().__init__(
            input_dim=input_dim, latent_dim=latent_dim, observation_dim=observation_dim
        )
        if quadrature is None:
            with tf.init_scope():
                quadrature = NDiagGHQuadrature(
                    self._quadrature_dim, DEFAULT_NUM_GAUSS_HERMITE_POINTS
                )
        self.quadrature = quadrature

    @property
    def _quadrature_dim(self) -> int:
        """
        This defines the number of dimensions over which to evaluate the
        quadrature. Generally, this is equal to self.latent_dim. This exists
        as a separate property to allow the ScalarLikelihood subclass to
        override it with 1 (broadcasting over observation/latent dimensions
        instead).
        """
        assert self.latent_dim is not None
        return self.latent_dim

    @check_shapes(
        "F: [broadcast batch..., latent_dim]",
        "X: [broadcast batch..., input_dim]",
        "Y: [broadcast batch..., observation_dim]",
        "return: [batch..., d]",
    )
    def _quadrature_log_prob(self, F: TensorType, X: TensorType, Y: TensorType) -> tf.Tensor:
        """
        Returns the appropriate log prob integrand for quadrature.

        Quadrature expects f(X), here logp(F), to return shape [N_quad_points]
        + batch_shape + [d']. Here d'=1, but log_prob() only returns
        [N_quad_points] + batch_shape, so we add an extra dimension.

        Also see _quadrature_reduction.
        """
        return tf.expand_dims(self.log_prob(X, F, Y), axis=-1)

    @check_shapes(
        "quadrature_result: [batch..., d]",
        "return: [batch...]",
    )
    def _quadrature_reduction(self, quadrature_result: TensorType) -> tf.Tensor:
        """
        Converts the quadrature integral appropriately.

        The return shape of quadrature is batch_shape + [d']. Here, d'=1, but
        we want predict_log_density and variational_expectations to return just
        batch_shape, so we squeeze out the extra dimension.

        Also see _quadrature_log_prob.
        """
        return tf.squeeze(quadrature_result, axis=-1)

    @inherit_check_shapes
    def _predict_log_density(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType
    ) -> tf.Tensor:
        r"""
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        :param X: input tensor
        :param Fmu: mean function evaluation tensor
        :param Fvar: variance of function evaluation tensor
        :param Y: observation tensor
        :returns: log predictive density
        """
        return self._quadrature_reduction(
            self.quadrature.logspace(self._quadrature_log_prob, Fmu, Fvar, X=X, Y=Y)
        )

    @inherit_check_shapes
    def _variational_expectations(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType
    ) -> tf.Tensor:
        r"""
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        :param X: input tensor
        :param Fmu: mean function evaluation tensor
        :param Fvar: variance of function evaluation tensor
        :param Y: observation tensor
        :returns: variational expectations
        """
        return self._quadrature_reduction(
            self.quadrature(self._quadrature_log_prob, Fmu, Fvar, X=X, Y=Y)
        )

    @inherit_check_shapes
    def _predict_mean_and_var(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType
    ) -> MeanAndVariance:
        r"""
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (e.g. Gaussian) will implement specific cases.

        :param X: input tensor
        :param Fmu: mean function evaluation tensor
        :param Fvar: variance of function evaluation tensor
        :returns: mean and variance of Y
        """

        def conditional_mean(F: TensorType, X_: TensorType) -> tf.Tensor:
            return self.conditional_mean(X_, F)

        def conditional_y_squared(F: TensorType, X_: TensorType) -> tf.Tensor:
            return self.conditional_variance(X_, F) + tf.square(self.conditional_mean(X_, F))

        E_y, E_y2 = self.quadrature([conditional_mean, conditional_y_squared], Fmu, Fvar, X_=X)
        V_y = E_y2 - E_y ** 2
        return E_y, V_y


class ScalarLikelihood(QuadratureLikelihood, abc.ABC):
    """
    A likelihood class that helps with scalar likelihood functions: likelihoods where
    each scalar latent function is associated with a single scalar observation variable.

    If there are multiple latent functions, then there must be a corresponding number of data: we
    check for this.

    The `Likelihood` class contains methods to compute marginal statistics of functions
    of the latents and the data ϕ(y,x,f):

    * variational_expectations:  ϕ(y,x,f) = log p(y|x,f)
    * predict_log_density: ϕ(y,x,f) = p(y|x,f)

    Those statistics are computed after having first marginalized the latent processes f
    under a multivariate normal distribution q(x,f) that is fully factorized.

    Some univariate integrals can be done by quadrature: we implement quadrature routines for 1D
    integrals in this class, though they may be overwritten by inheriting classes where those
    integrals are available in closed form.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(input_dim=None, latent_dim=None, observation_dim=None, **kwargs)

    @inherit_check_shapes
    def _log_prob(self, X: TensorType, F: TensorType, Y: TensorType) -> tf.Tensor:
        r"""
        Compute log p(Y|X,F), where by convention we sum out the last axis as it represented
        independent latent functions and observations.
        :param F: function evaluation tensor
        :param Y: observation tensor
        """
        return tf.reduce_sum(self._scalar_log_prob(X, F, Y), axis=-1)

    @abc.abstractmethod
    @check_shapes(
        "X: [broadcast batch..., input_dim]",
        "F: [broadcast batch..., latent_dim]",
        "Y: [broadcast batch..., observation_dim]",
        "return: [batch..., latent_dim]",
    )
    def _scalar_log_prob(self, X: TensorType, F: TensorType, Y: TensorType) -> tf.Tensor:
        raise NotImplementedError

    @property
    def _quadrature_dim(self) -> int:
        """
        Quadrature is over the latent dimensions. Generally, this is equal to
        self.latent_dim. This separate property allows the ScalarLikelihood
        subclass to override it with 1 (broadcasting over observation/latent
        dimensions instead).
        """
        return 1

    @inherit_check_shapes
    def _quadrature_log_prob(self, F: TensorType, X: TensorType, Y: TensorType) -> tf.Tensor:
        """
        Returns the appropriate log prob integrand for quadrature.

        Quadrature expects f(X), here logp(F), to return shape [N_quad_points]
        + batch_shape + [d']. Here d' corresponds to the last dimension of both
        F and Y, and _scalar_log_prob simply broadcasts over this.

        Also see _quadrature_reduction.
        """
        return self._scalar_log_prob(X, F, Y)

    @inherit_check_shapes
    def _quadrature_reduction(self, quadrature_result: TensorType) -> tf.Tensor:
        """
        Converts the quadrature integral appropriately.

        The return shape of quadrature is batch_shape + [d']. Here, d'
        corresponds to the last dimension of both F and Y, and we want to sum
        over the observations to obtain the overall predict_log_density or
        variational_expectations.

        Also see _quadrature_log_prob.
        """
        return tf.reduce_sum(quadrature_result, axis=-1)


class SwitchedLikelihood(ScalarLikelihood):
    def __init__(self, likelihood_list: Iterable[ScalarLikelihood], **kwargs: Any) -> None:
        """
        In this likelihood, we assume at extra column of Y, which contains
        integers that specify a likelihood from the list of likelihoods.
        """
        super().__init__(**kwargs)
        self.likelihoods = list(likelihood_list)

    @check_shapes(
        "args[all]: [batch..., .]",
        "return: [batch..., ...]",
    )
    def _partition_and_stitch(self, args: Sequence[TensorType], func_name: str) -> tf.Tensor:
        """
        args is a list of tensors, to be passed to self.likelihoods.<func_name>

        args[-1] is the 'Y' argument, which contains the indexes to self.likelihoods.

        This function splits up the args using dynamic_partition, calls the
        relevant function on the likelihoods, and re-combines the result.
        """
        # get the index from Y
        args_list = list(args)
        Y = args_list[-1]
        ind = Y[..., -1]
        ind = tf.cast(ind, tf.int32)
        Y = Y[..., :-1]
        args_list[-1] = Y

        # split up the arguments into chunks corresponding to the relevant likelihoods
        args_chunks = zip(*[tf.dynamic_partition(X, ind, len(self.likelihoods)) for X in args_list])

        # apply the likelihood-function to each section of the data
        funcs = [getattr(lik, func_name) for lik in self.likelihoods]
        results = [f(*args_i) for f, args_i in zip(funcs, args_chunks)]

        # stitch the results back together
        partitions = tf.dynamic_partition(tf.range(0, tf.size(ind)), ind, len(self.likelihoods))
        results = tf.dynamic_stitch(partitions, results)

        return results

    @inherit_check_shapes
    def _scalar_log_prob(self, X: TensorType, F: TensorType, Y: TensorType) -> tf.Tensor:
        return self._partition_and_stitch([X, F, Y], "_scalar_log_prob")

    @inherit_check_shapes
    def _predict_log_density(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType
    ) -> tf.Tensor:
        return self._partition_and_stitch([X, Fmu, Fvar, Y], "predict_log_density")

    @inherit_check_shapes
    def _variational_expectations(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType
    ) -> tf.Tensor:
        return self._partition_and_stitch([X, Fmu, Fvar, Y], "variational_expectations")

    @inherit_check_shapes
    def _predict_mean_and_var(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType
    ) -> MeanAndVariance:
        mvs = [lik.predict_mean_and_var(X, Fmu, Fvar) for lik in self.likelihoods]
        mu_list, var_list = zip(*mvs)
        mu = tf.concat(mu_list, axis=1)
        var = tf.concat(var_list, axis=1)
        return mu, var

    @inherit_check_shapes
    def _conditional_mean(self, X: TensorType, F: TensorType) -> tf.Tensor:
        raise NotImplementedError

    @inherit_check_shapes
    def _conditional_variance(self, X: TensorType, F: TensorType) -> tf.Tensor:
        raise NotImplementedError


class MonteCarloLikelihood(Likelihood):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.num_monte_carlo_points = 100

    @check_shapes(
        "Fmu: [batch..., latent_dim]",
        "Fvar: [batch..., latent_dim]",
        "Ys.values(): [batch..., .]",
        "return: [broadcast n_funcs, batch..., .]",
    )
    def _mc_quadrature(
        self,
        funcs: Union[Callable[..., tf.Tensor], Iterable[Callable[..., tf.Tensor]]],
        Fmu: TensorType,
        Fvar: TensorType,
        logspace: bool = False,
        epsilon: Optional[TensorType] = None,
        **Ys: TensorType,
    ) -> tf.Tensor:
        return ndiag_mc(funcs, self.num_monte_carlo_points, Fmu, Fvar, logspace, epsilon, **Ys)

    @inherit_check_shapes
    def _predict_mean_and_var(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType, epsilon: Optional[TensorType] = None
    ) -> MeanAndVariance:
        r"""
        Given a Normal distribution for the latent function,
        return the mean of Y

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive mean

           ∫∫ y p(y|f)q(f) df dy

        and the predictive variance

           ∫∫ y² p(y|f)q(f) df dy  - [ ∫∫ y p(y|f)q(f) df dy ]²

        Here, we implement a default Monte Carlo routine.
        """

        def conditional_mean(F: TensorType, X_: TensorType) -> TensorType:
            return self.conditional_mean(X_, F)

        def conditional_y_squared(F: TensorType, X_: TensorType) -> TensorType:
            return self.conditional_variance(X_, F) + tf.square(self.conditional_mean(X_, F))

        E_y, E_y2 = self._mc_quadrature(
            [conditional_mean, conditional_y_squared],
            Fmu,
            Fvar,
            epsilon=epsilon,
            X_=X,
        )
        V_y = E_y2 - tf.square(E_y)
        return E_y, V_y  # [N, D]

    @inherit_check_shapes
    def _predict_log_density(
        self,
        X: TensorType,
        Fmu: TensorType,
        Fvar: TensorType,
        Y: TensorType,
        epsilon: Optional[TensorType] = None,
    ) -> tf.Tensor:
        r"""
        Given a Normal distribution for the latent function, and a datum Y,
        compute the log predictive density of Y.

        i.e. if
            q(x, f) = N(Fmu, Fvar)

        and this object represents

            p(y|x,f)

        then this method computes the predictive density

            log ∫ p(y=Y|x,f)q(x,f) df

        Here, we implement a default Monte Carlo routine.
        """

        def log_prob(F: TensorType, X_: TensorType, Y_: TensorType) -> tf.Tensor:
            return self.log_prob(X_, F, Y_)

        return tf.reduce_sum(
            self._mc_quadrature(log_prob, Fmu, Fvar, logspace=True, epsilon=epsilon, X_=X, Y_=Y),
            axis=-1,
        )

    @inherit_check_shapes
    def _variational_expectations(
        self,
        X: TensorType,
        Fmu: TensorType,
        Fvar: TensorType,
        Y: TensorType,
        epsilon: Optional[TensorType] = None,
    ) -> tf.Tensor:
        r"""
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values.

        if
            q(x,f) = N(Fmu, Fvar)  - Fmu: [N, D]  Fvar: [N, D]

        and this object represents

            p(y|x,f)  - Y: [N, 1]

        then this method computes

           ∫ (log p(y|x,f)) q(x,f) df.


        Here, we implement a default Monte Carlo quadrature routine.
        """

        def log_prob(F: TensorType, X_: TensorType, Y_: TensorType) -> tf.Tensor:
            return self.log_prob(X_, F, Y_)

        return tf.reduce_sum(
            self._mc_quadrature(log_prob, Fmu, Fvar, epsilon=epsilon, X_=X, Y_=Y), axis=-1
        )
