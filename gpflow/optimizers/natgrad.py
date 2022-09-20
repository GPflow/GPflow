# Copyright 2018-2020 The GPflow Contributors. All Rights Reserved.
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
import functools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

from ..base import Parameter, _to_constrained
from ..experimental.check_shapes import check_shapes

Scalar = Union[float, tf.Tensor, np.ndarray]
LossClosure = Callable[[], tf.Tensor]
NatGradParameters = Union[Tuple[Parameter, Parameter], Tuple[Parameter, Parameter, "XiTransform"]]

__all__ = [
    "NaturalGradient",
    "XiNat",
    "XiSqrtMeanVar",
    "XiTransform",
]


#
# Xi transformations necessary for natural gradient optimizer.
# Abstract class and two implementations: XiNat and XiSqrtMeanVar.
#


class XiTransform(metaclass=abc.ABCMeta):
    """
    XiTransform is the base class that implements three transformations necessary
    for the natural gradient calculation wrt any parameterization.
    """

    @staticmethod
    @abc.abstractmethod
    @check_shapes(
        "mean: [N, D]",
        "varsqrt: [D, N, N]",
        "return[0]: [N, D]",
        "return[1]: [D, N, N]",
    )
    def meanvarsqrt_to_xi(mean: tf.Tensor, varsqrt: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Transforms the parameter `mean` and `varsqrt` to `xi1`, `xi2`

        :param mean: the mean parameter
        :param varsqrt: the varsqrt parameter
        :return: tuple (xi1, xi2), the xi parameters
        """

    @staticmethod
    @abc.abstractmethod
    @check_shapes(
        "xi1: [N, D]",
        "xi2: [D, N, N]",
        "return[0]: [N, D]",
        "return[1]: [D, N, N]",
    )
    def xi_to_meanvarsqrt(xi1: tf.Tensor, xi2: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Transforms the parameter `xi1`, `xi2` to `mean`, `varsqrt`

        :param xi1: the ξ₁ parameter
        :param xi2: the ξ₂ parameter
        :return: tuple (mean, varsqrt), the meanvarsqrt parameters
        """

    @staticmethod
    @abc.abstractmethod
    @check_shapes(
        "nat1: [N, D]",
        "nat2: [D, N, N]",
        "return[0]: [N, D]",
        "return[1]: [D, N, N]",
    )
    def naturals_to_xi(nat1: tf.Tensor, nat2: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Applies the transform so that `nat1`, `nat2` is mapped to `xi1`, `xi2`

        :param nat1: the θ₁ parameter
        :param nat2: the θ₂ parameter
        :return: tuple `xi1`, `xi2`
        """


class XiNat(XiTransform):
    """
    This is the default transform. Using the natural directly saves the forward mode
    gradient, and also gives the analytic optimal solution for gamma=1 in the case
    of Gaussian likelihood.
    """

    @staticmethod
    @check_shapes(
        "mean: [N, D]",
        "varsqrt: [D, N, N]",
        "return[0]: [N, D]",
        "return[1]: [D, N, N]",
    )
    def meanvarsqrt_to_xi(mean: tf.Tensor, varsqrt: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return meanvarsqrt_to_natural(mean, varsqrt)

    @staticmethod
    @check_shapes(
        "xi1: [N, D]",
        "xi2: [D, N, N]",
        "return[0]: [N, D]",
        "return[1]: [D, N, N]",
    )
    def xi_to_meanvarsqrt(xi1: tf.Tensor, xi2: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return natural_to_meanvarsqrt(xi1, xi2)

    @staticmethod
    @check_shapes(
        "nat1: [N, D]",
        "nat2: [D, N, N]",
        "return[0]: [N, D]",
        "return[1]: [D, N, N]",
    )
    def naturals_to_xi(nat1: tf.Tensor, nat2: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return nat1, nat2


class XiSqrtMeanVar(XiTransform):
    """
    This transformation will perform natural gradient descent on the model parameters,
    so saves the conversion to and from Xi.
    """

    @staticmethod
    @check_shapes(
        "mean: [N, D]",
        "varsqrt: [D, N, N]",
        "return[0]: [N, D]",
        "return[1]: [D, N, N]",
    )
    def meanvarsqrt_to_xi(mean: tf.Tensor, varsqrt: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return mean, varsqrt

    @staticmethod
    @check_shapes(
        "xi1: [N, D]",
        "xi2: [D, N, N]",
        "return[0]: [N, D]",
        "return[1]: [D, N, N]",
    )
    def xi_to_meanvarsqrt(xi1: tf.Tensor, xi2: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return xi1, xi2

    @staticmethod
    @check_shapes(
        "nat1: [N, D]",
        "nat2: [D, N, N]",
        "return[0]: [N, D]",
        "return[1]: [D, N, N]",
    )
    def naturals_to_xi(nat1: tf.Tensor, nat2: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return natural_to_meanvarsqrt(nat1, nat2)


class NaturalGradient(tf.optimizers.Optimizer):
    """
    Implements a natural gradient descent optimizer for variational models
    that are based on a distribution q(u) = N(q_mu, q_sqrt q_sqrtᵀ) that is
    parameterized by mean q_mu and lower-triangular Cholesky factor q_sqrt
    of the covariance.

    Note that this optimizer does not implement the standard API of
    tf.optimizers.Optimizer. Its only public method is minimize(), which has
    a custom signature (var_list needs to be a list of (q_mu, q_sqrt) tuples,
    where q_mu and q_sqrt are gpflow.Parameter instances, not tf.Variable).

    Note furthermore that the natural gradients are implemented only for the
    full covariance case (i.e., q_diag=True is NOT supported).

    When using in your work, please cite :cite:t:`salimbeni18`.
    """

    def __init__(
        self, gamma: Scalar, xi_transform: XiTransform = XiNat(), name: Optional[str] = None
    ) -> None:
        """
        :param gamma: natgrad step length
        :param xi_transform: default ξ transform (can be overridden in the call to minimize())
            The XiNat default choice works well in general.
        """
        name = self.__class__.__name__ if name is None else name
        super().__init__(name)
        self.gamma = gamma
        self.xi_transform = xi_transform

    @check_shapes(
        "var_list[all][0]: [N, D]",
        "var_list[all][1]: [D, N, N]",
    )
    def minimize(
        self,
        loss_fn: LossClosure,
        var_list: Sequence[NatGradParameters],
    ) -> None:
        """
        Minimizes objective function of the model.
        Natural Gradient optimizer works with variational parameters only.

        GPflow implements the `XiNat` (default) and `XiSqrtMeanVar` transformations
        for parameters. Custom transformations that implement the `XiTransform`
        interface are also possible.

        :param loss_fn: Loss function.
        :param var_list: List of pair tuples of variational parameters or
            triplet tuple with variational parameters and ξ transformation.
            If ξ is not specified, will use self.xi_transform.
            For example, `var_list` could be::

                var_list = [
                    (q_mu1, q_sqrt1),
                    (q_mu2, q_sqrt2, XiSqrtMeanVar())
                ]
        """
        parameters = [(v[0], v[1], (v[2] if len(v) > 2 else None)) for v in var_list]  # type: ignore[misc]
        self._natgrad_steps(loss_fn, parameters)

    @check_shapes(
        "parameters[all][0]: [N, D]",
        "parameters[all][1]: [D, N, N]",
    )
    def _natgrad_steps(
        self,
        loss_fn: LossClosure,
        parameters: Sequence[Tuple[Parameter, Parameter, Optional[XiTransform]]],
    ) -> None:
        """
        Computes gradients of loss_fn() w.r.t. q_mu and q_sqrt, and updates
        these parameters using the natgrad backwards step, for all sets of
        variational parameters passed in.

        :param loss_fn: Loss function.
        :param parameters: List of tuples (q_mu, q_sqrt, xi_transform)
        """
        q_mus, q_sqrts, xis = zip(*parameters)
        q_mu_vars = [p.unconstrained_variable for p in q_mus]
        q_sqrt_vars = [p.unconstrained_variable for p in q_sqrts]

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(q_mu_vars + q_sqrt_vars)
            loss = loss_fn()

        q_mu_grads, q_sqrt_grads = tape.gradient(loss, [q_mu_vars, q_sqrt_vars])
        # NOTE that these are the gradients in *unconstrained* space

        with tf.name_scope(f"{self._name}/natural_gradient_steps"):
            for q_mu_grad, q_sqrt_grad, q_mu, q_sqrt, xi_transform in zip(
                q_mu_grads, q_sqrt_grads, q_mus, q_sqrts, xis
            ):
                self._natgrad_apply_gradients(q_mu_grad, q_sqrt_grad, q_mu, q_sqrt, xi_transform)

    @check_shapes(
        "q_mu_grad: [N, D]",
        "q_sqrt_grad: [D, N_N_transformed...]",
        "q_mu: [N, D]",
        "q_sqrt: [D, N, N]",
    )
    def _natgrad_apply_gradients(
        self,
        q_mu_grad: tf.Tensor,
        q_sqrt_grad: tf.Tensor,
        q_mu: Parameter,
        q_sqrt: Parameter,
        xi_transform: Optional[XiTransform] = None,
    ) -> None:
        """
        This function does the backward step on the q_mu and q_sqrt parameters,
        given the gradients of the loss function with respect to their unconstrained
        variables. I.e., it expects the arguments to come from

            with tf.GradientTape() as tape:
                loss = loss_function()
            q_mu_grad, q_mu_sqrt = tape.gradient(loss, [q_mu, q_sqrt])

        (Note that tape.gradient() returns the gradients in *unconstrained* space!)

        Implements equation [10] from :cite:t:`salimbeni18`.

        In addition, for convenience with the rest of GPflow, this code computes ∂L/∂η using
        the chain rule (the following assumes a numerator layout where the gradient is a row
        vector; note that TensorFlow actually returns a column vector), where L is the loss:

        ∂L/∂η = (∂L / ∂[q_mu, q_sqrt])(∂[q_mu, q_sqrt] / ∂η)

        In total there are three derivative calculations:
        natgrad of L w.r.t ξ  = (∂ξ / ∂θ) [(∂L / ∂[q_mu, q_sqrt]) (∂[q_mu, q_sqrt] / ∂η)]ᵀ

        Note that if ξ = θ (i.e. [q_mu, q_sqrt]) some of these calculations are the identity.
        In the code η = eta, ξ = xi, θ = nat.

        :param q_mu_grad: gradient of loss w.r.t. q_mu (in unconstrained space)
        :param q_sqrt_grad: gradient of loss w.r.t. q_sqrt (in unconstrained space)
        :param q_mu: parameter for the mean of q(u) with shape [M, L]
        :param q_sqrt: parameter for the square root of the covariance of q(u)
            with shape [L, M, M] (the diagonal parametrization, q_diag=True, is NOT supported)
        :param xi_transform: the ξ transform to use (self.xi_transform if not specified)
        """
        if xi_transform is None:
            xi_transform = self.xi_transform

        # 1) the ordinary gpflow gradient
        dL_dmean = _to_constrained(q_mu_grad, q_mu.transform)
        dL_dvarsqrt = _to_constrained(q_sqrt_grad, q_sqrt.transform)

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch([q_mu.unconstrained_variable, q_sqrt.unconstrained_variable])

            # the three parameterizations as functions of [q_mu, q_sqrt]
            eta1, eta2 = meanvarsqrt_to_expectation(q_mu, q_sqrt)
            # we need these to calculate the relevant gradients
            meanvarsqrt = expectation_to_meanvarsqrt(eta1, eta2)

            if not isinstance(xi_transform, XiNat):
                nat1, nat2 = meanvarsqrt_to_natural(q_mu, q_sqrt)
                xi1_nat, xi2_nat = xi_transform.naturals_to_xi(nat1, nat2)
                dummy_tensors = tf.ones_like(xi1_nat), tf.ones_like(xi2_nat)
                with tf.GradientTape(watch_accessed_variables=False) as forward_tape:
                    forward_tape.watch(dummy_tensors)
                    dummy_gradients = tape.gradient(
                        [xi1_nat, xi2_nat], [nat1, nat2], output_gradients=dummy_tensors
                    )

        # 2) the chain rule to get ∂L/∂η, where η (eta) are the expectation parameters
        dL_deta1, dL_deta2 = tape.gradient(
            meanvarsqrt, [eta1, eta2], output_gradients=[dL_dmean, dL_dvarsqrt]
        )

        if not isinstance(xi_transform, XiNat):
            nat_dL_xi1, nat_dL_xi2 = forward_tape.gradient(
                dummy_gradients, dummy_tensors, output_gradients=[dL_deta1, dL_deta2]
            )
        else:
            nat_dL_xi1, nat_dL_xi2 = dL_deta1, dL_deta2

        del tape  # Remove "persistent" tape

        xi1, xi2 = xi_transform.meanvarsqrt_to_xi(q_mu, q_sqrt)
        xi1_new = xi1 - self.gamma * nat_dL_xi1
        xi2_new = xi2 - self.gamma * nat_dL_xi2

        # Transform back to the model parameters [q_mu, q_sqrt]
        mean_new, varsqrt_new = xi_transform.xi_to_meanvarsqrt(xi1_new, xi2_new)

        q_mu.assign(mean_new)
        q_sqrt.assign(varsqrt_new)

    def get_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = super().get_config()
        config.update({"gamma": self._serialize_hyperparameter("gamma")})
        return config


#
# Auxiliary gaussian parameter conversion functions.
#
# The following functions expect their first and second inputs to have shape
# [D, N, 1] and [D, N, N], respectively. Return values are also of shapes [D, N, 1] and [D, N, N].


def swap_dimensions(
    method: Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]
) -> Callable[..., Tuple[tf.Tensor, tf.Tensor]]:
    """
    Converts between GPflow indexing and tensorflow indexing
    `method` is a function that broadcasts over the first dimension (i.e. like all tensorflow matrix
    ops):

    * `method` inputs [D, N, 1], [D, N, N]
    * `method` outputs [D, N, 1], [D, N, N]

    :return: Function that broadcasts over the final dimension (i.e. compatible with GPflow):

        * inputs: [N, D], [D, N, N]
        * outputs: [N, D], [D, N, N]
    """

    @functools.wraps(method)
    @check_shapes(
        "a_nd: [N, D] if swap",
        "a_nd: [D, N, 1] if not swap",
        "b_dnn: [D, N, N]",
        "return[0]: [N, D] if swap",
        "return[0]: [D, N, 1] if not swap",
        "return[1]: [D, N, N]",
    )
    def wrapper(
        a_nd: tf.Tensor, b_dnn: tf.Tensor, swap: bool = True
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        if swap:
            a_dn1 = tf.linalg.adjoint(a_nd)[:, :, None]
            A_dn1, B_dnn = method(a_dn1, b_dnn)
            A_nd = tf.linalg.adjoint(A_dn1[:, :, 0])
            return A_nd, B_dnn
        else:
            return method(a_nd, b_dnn)

    return wrapper


@swap_dimensions
@check_shapes(
    "nat1: [D, N, 1]",
    "nat2: [D, N, N]",
    "return[0]: [D, N, 1]",
    "return[1]: [D, N, N]",
)
def natural_to_meanvarsqrt(nat1: tf.Tensor, nat2: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    var_sqrt_inv = tf.linalg.cholesky(-2 * nat2)
    var_sqrt = _inverse_lower_triangular(var_sqrt_inv)
    S = tf.linalg.matmul(var_sqrt, var_sqrt, transpose_a=True)
    mu = tf.linalg.matmul(S, nat1)
    # We need the decomposition of S as L L^T, not as L^T L,
    # hence we need another cholesky.
    return mu, tf.linalg.cholesky(S)


@swap_dimensions
@check_shapes(
    "mu: [D, N, 1]",
    "s_sqrt: [D, N, N]",
    "return[0]: [D, N, 1]",
    "return[1]: [D, N, N]",
)
def meanvarsqrt_to_natural(mu: tf.Tensor, s_sqrt: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    s_sqrt_inv = _inverse_lower_triangular(s_sqrt)
    s_inv = tf.linalg.matmul(s_sqrt_inv, s_sqrt_inv, transpose_a=True)
    return tf.linalg.matmul(s_inv, mu), -0.5 * s_inv


@swap_dimensions
@check_shapes(
    "nat1: [D, N, 1]",
    "nat2: [D, N, N]",
    "return[0]: [D, N, 1]",
    "return[1]: [D, N, N]",
)
def natural_to_expectation(nat1: tf.Tensor, nat2: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    args = natural_to_meanvarsqrt(nat1, nat2, swap=False)
    return meanvarsqrt_to_expectation(*args, swap=False)


@swap_dimensions
@check_shapes(
    "eta1: [D, N, 1]",
    "eta2: [D, N, N]",
    "return[0]: [D, N, 1]",
    "return[1]: [D, N, N]",
)
def expectation_to_natural(eta1: tf.Tensor, eta2: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    args = expectation_to_meanvarsqrt(eta1, eta2, swap=False)
    return meanvarsqrt_to_natural(*args, swap=False)


@swap_dimensions
@check_shapes(
    "eta1: [D, N, 1]",
    "eta2: [D, N, N]",
    "return[0]: [D, N, 1]",
    "return[1]: [D, N, N]",
)
def expectation_to_meanvarsqrt(eta1: tf.Tensor, eta2: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    var = eta2 - tf.linalg.matmul(eta1, eta1, transpose_b=True)
    return eta1, tf.linalg.cholesky(var)


@swap_dimensions
@check_shapes(
    "m: [D, N, 1]",
    "v_sqrt: [D, N, N]",
    "return[0]: [D, N, 1]",
    "return[1]: [D, N, N]",
)
def meanvarsqrt_to_expectation(m: tf.Tensor, v_sqrt: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    v = tf.linalg.matmul(v_sqrt, v_sqrt, transpose_b=True)
    return m, v + tf.linalg.matmul(m, m, transpose_b=True)


@check_shapes(
    "M: [D, N, N]",
    "return: [D, N, N]",
)
def _inverse_lower_triangular(M: tf.Tensor) -> tf.Tensor:
    """
    Take inverse of lower triangular (e.g. Cholesky) matrix. This function
    broadcasts over the first index.

    :param M: Tensor with lower triangular structure of shape [D, N, N]
    :return: The inverse of the Cholesky decomposition. Same shape as input.
    """
    if M.shape.ndims != 3:  # pragma: no cover
        raise ValueError("Number of dimensions for input is required to be 3.")
    D, N = tf.shape(M)[0], tf.shape(M)[1]
    I_dnn = tf.eye(N, dtype=M.dtype)[None, :, :] * tf.ones((D, 1, 1), dtype=M.dtype)
    return tf.linalg.triangular_solve(M, I_dnn)
