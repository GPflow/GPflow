# Copyright 2018 Hugh Salimbeni, Artem Artemev @awav
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

import tensorflow as tf

from . import optimizer
from .. import settings
from ..actions import Optimization
from ..models import Model


class NatGradOptimizer(optimizer.Optimizer):
    def __init__(self, gamma, **kwargs):
        super().__init__(**kwargs)
        self.name = self.__class__.__name__
        self._gamma = gamma
        self._natgrad_op = None

    @property
    def gamma(self):
        return self._gamma

    @property
    def minimize_operation(self):
        return self._natgrad_op

    def minimize(self, model, var_list=None, session=None, feed_dict=None,
                 maxiter=1000, anchor=True, step_callback=None, **kwargs):
        """
        Minimizes objective function of the model.
        Natural Gradient optimizer works with variational parameters only.
        There are two supported ways of transformation for parameters:
            - XiNat
            - XiSqrtMeanVar
        Custom transformations are also possible, they should implement
        `XiTransform` interface.

            :param model: GPflow model.
            :param session: Tensorflow session where optimization will be run.
            :param var_list: List of pair tuples of variational parameters or
                triplet tuple with variational parameters and ξ transformation.
                By default, all parameters goes through XiNat() transformation.
                For example your `var_list` can look as,
                ```
                var_list = [
                    (q_mu1, q_sqrt1),
                    (q_mu2, q_sqrt2, XiSqrtMeanVar())
                ]
                ```
            :param feed_dict: Feed dictionary of tensors passed to session run method.
            :param maxiter: Number of run interation. Default value: 1000.
            :param anchor: Synchronize updated parameters for a session with internal
                parameter's values.
            :param step_callback: A callback function to execute at each optimization step.
                The callback should accept variable argument list, where first argument is
                optimization step number.
            :type step_callback: Callable[[], None]
            :param kwargs: Extra parameters passed to session run's method.
        """

        if model is None or not isinstance(model, Model):
            raise ValueError('Unknown type passed for optimization.')

        self._model = model
        session = model.enquire_session(session)
        opt = self.make_optimize_action(model, session=session, var_list=var_list, **kwargs)
        with session.as_default():
            for step in range(maxiter):
                opt()
                if step_callback is not None:
                    step_callback(step)
        if anchor:
            model.anchor(session)

    def make_optimize_tensor(self, model, session=None, var_list=None):
        """
        Make Tensorflow optimization tensor.
        This method builds natural gradients optimization tensor and initializes all
        necessary variables created by the optimizer.

            :param model: GPflow model.
            :param session: Tensorflow session.
            :param var_list: List of tuples of variational parameters.
            :return: Tensorflow natural gradient operation.
        """
        session = model.enquire_session(session)
        with session.as_default(), tf.name_scope(self.name):
            # Create optimizer variables before initialization.
            return self._build_natgrad_step_ops(model, *var_list)

    def make_optimize_action(self, model, session=None, var_list=None, **kwargs):
        """
        Builds optimization action.
        Natural Gradient optimizer works with variational parameters only.
        There are two supported ways of transformation for parameters:
            - XiNat
            - XiSqrtMeanVar
        Custom transformations are also possible, they should implement
        `XiTransform` interface.

            :param model: GPflow model.
            :param session: Tensorflow session where optimization will be run.
            :param var_list: List of pair tuples of variational parameters or
                triplet tuple with variational parameters and ξ transformation.
                By default, all parameters goes through XiNat() transformation.
                For example your `var_list` can look as,
                ```
                var_list = [
                    (q_mu1, q_sqrt1),
                    (q_mu2, q_sqrt2, XiSqrtMeanVar())
                ]
                ```
            :param kwargs: Extra parameters passed to session's run method.
            :return: Optimization action.
        """
        if model is None or not isinstance(model, Model):
            raise ValueError('Unknown type passed for optimization.')
        feed_dict = kwargs.pop('feed_dict', None)
        feed_dict_update = self._gen_feed_dict(model, feed_dict)
        run_kwargs = {} if feed_dict_update is None else {'feed_dict': feed_dict_update}
        optimizer_tensor = self.make_optimize_tensor(model, session=session, var_list=var_list)
        opt = Optimization()
        opt.with_optimizer(self)
        opt.with_model(model)
        opt.with_optimizer_tensor(optimizer_tensor)
        opt.with_run_kwargs(**run_kwargs)
        return opt

    @staticmethod
    def _forward_gradients(ys, xs, d_xs):
        """
        Forward-mode pushforward analogous to the pullback defined by tf.gradients.
        With tf.gradients, grad_ys is the vector being pulled back, and here d_xs is
        the vector being pushed forward, i.e. this computes (∂ys / ∂xs)^T ∂xs.

        This is adapted from https://github.com/HIPS/autograd/pull/175#issuecomment-306984338

        :param ys: list of variables being differentiated (tensor)
        :param xs: list of variables to differentiate wrt (tensor)
        :param d_xs: list of gradients to push forward (same shapes as ys)
        :return: the specified moment of the variational distribution
        """
        # this should be v = [tf.placeholder(y.dtype) for y in ys], but tensorflow
        # wants a value for the placeholder, even though it never gets used
        v = [tf.placeholder_with_default(tf.zeros(y.get_shape(), dtype=y.dtype),
                                         shape=y.get_shape()) for y in ys]

        g = tf.gradients(ys, xs, grad_ys=v)
        return tf.gradients(g, v, grad_ys=d_xs)

    def _build_natgrad_step_ops(self, model, *args):
        ops = []
        for arg in args:
            q_mu, q_sqrt = arg[:2]
            xi_transform = arg[2] if len(arg) > 2 else XiNat()
            ops.append(self._build_natgrad_step_op(model, q_mu, q_sqrt, xi_transform))
        ops = list(sum(ops, ()))
        return tf.group(*ops)

    def _build_natgrad_step_op(self, model, q_mu_param, q_sqrt_param, xi_transform):
        """
        Implements equation 10 from

        @inproceedings{salimbeni18,
            title={Natural Gradients in Practice: Non-Conjugate  Variational Inference in Gaussian Process Models},
            author={Salimbeni, Hugh and Eleftheriadis, Stefanos and Hensman, James},
            booktitle={AISTATS},
            year={2018}

        In addition, for convenience with the rest of GPflow, this code computes ∂L/∂η using
        the chain rule:

        ∂L/∂η = (∂[q_μ, q_sqrt] / ∂η)(∂L / ∂[q_μ, q_sqrt])

        In total there are three derivative calculations:
        natgrad L w.r.t ξ  = (∂ξ / ∂nat ) [ (∂[q_μ, q_sqrt] / ∂η)(∂L / ∂[q_μ, q_sqrt]) ]^T

        Note that if ξ = nat or [q_μ, q_sqrt] some of these calculations are the identity.

        """
        objective = model.objective
        q_mu, q_sqrt = q_mu_param.constrained_tensor, q_sqrt_param.constrained_tensor

        # the three parameterizations as functions of [q_mu, q_sqrt]
        etas = meanvarsqrt_to_expectation(q_mu, q_sqrt)
        nats = meanvarsqrt_to_natural(q_mu, q_sqrt)
        xis = xi_transform.meanvarsqrt_to_xi(q_mu, q_sqrt)

        # we need these to calculate the relevant gradients
        _meanvarsqrt = expectation_to_meanvarsqrt(*etas)

        ## three derivatives
        # 1) the oridinary gpflow gradient
        dL_d_mean, dL_d_varsqrt = tf.gradients(objective, [q_mu, q_sqrt])

        # 2) the chain rule to get ∂L/∂η, where eta are the expectation parameters
        dL_detas = tf.gradients(_meanvarsqrt, etas, grad_ys=[dL_d_mean, dL_d_varsqrt])

        # 3) the forward mode gradient to calculate (∂ξ / ∂nat)(∂L / ∂η)^T,
        if isinstance(xi_transform, XiNat):
            nat_dL_xis = dL_detas
        else:
            _xis = xi_transform.naturals_to_xi(*nats)
            # this line should be removed if the placeholder_with_default problem is rectified
            _xis = [tf.reshape(_xis[0], q_mu_param.shape), tf.reshape(_xis[1], q_sqrt_param.shape)]
            nat_dL_xis = self._forward_gradients(_xis, nats, dL_detas)

        # perform natural gradient descent on the ξ parameters
        xis_new = [xi - self.gamma * nat_dL_xi for xi, nat_dL_xi in zip(xis, nat_dL_xis)]

        # transform back to the model parameters [q_μ, q_sqrt]
        mean_new, varsqrt_new = xi_transform.xi_to_meanvarsqrt(*xis_new)

        # these are the tensorflow variables to assign
        q_mu_u = q_mu_param.unconstrained_tensor
        q_sqrt_u = q_sqrt_param.unconstrained_tensor

        # so the transform to work for LowerTriangular
        mean_new.set_shape(q_mu_param.shape)
        varsqrt_new.set_shape(q_sqrt_param.shape)

        q_mu_assign = tf.assign(q_mu_u, q_mu_param.transform.backward_tensor(mean_new))
        q_sqrt_assign = tf.assign(q_sqrt_u, q_sqrt_param.transform.backward_tensor(varsqrt_new))
        return q_mu_assign, q_sqrt_assign

#
# Xi transformations necessary for natural gradient optimizer.
# Abstract class and two implementations: XiNat and XiSqrtMeanVar.
#

class XiTransform(metaclass=abc.ABCMeta):
    """
    XiTransform is the base class that implements three transformations necessary
    for the natural gradient calculation wrt any parameterization.
    This class does not handle any shape information, but it is assumed that
    the parameters pairs are always of shape (N, D) and (D, N, N).
    """
    @abc.abstractmethod
    def meanvarsqrt_to_xi(self, mean, varsqrt):
        """
        Transforms the parameter `mean` and `varsqrt` to `xi_1`, `xi_2`

        :param mean: the mean parameter (N, D)
        :param varsqrt: the varsqrt parameter (N, N, D)
        :return: tuple (xi_1, xi_2), the xi parameters (N, D), (N, N, D)
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def xi_to_meanvarsqrt(self, xi_1, xi_2):
        """
        Transforms the parameter `xi_1`, `xi_2` to `mean`, `varsqrt`

        :param xi_1: the xi_1 parameter
        :param xi_2: the xi_2 parameter
        :return: tuple (mean, varsqrt), the meanvarsqrt parameters
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def naturals_to_xi(self, nat_1, nat_2):
        """
        Applies the transform so that `nat_1`, `nat_2` is mapped to `xi_1`, `xi_2`

        :param nat_1: the nat_1 parameter
        :param nat_2: the nat_1 parameter
        :return: tuple `xi_1`, `xi_2`
        """
        pass  # pragma: no cover


class XiNat(XiTransform):
    """
    This is the default transform. Using the natural directly saves the forward mode
     gradient, and also gives the analytic optimal solution for gamma=1 in the case
     of Gaussian likelihood.
    """
    def meanvarsqrt_to_xi(self, mean, varsqrt):
        return meanvarsqrt_to_natural(mean, varsqrt)

    def xi_to_meanvarsqrt(self, xi_1, xi_2):
        return natural_to_meanvarsqrt(xi_1, xi_2)

    def naturals_to_xi(self, nat_1, nat_2):
        return nat_1, nat_2


class XiSqrtMeanVar(XiTransform):
    """
    This transformation will perform natural gradient descent on the model parameters,
    so saves the conversion to and from Xi.
    """
    def meanvarsqrt_to_xi(self, mean, varsqrt):
        return mean, varsqrt

    def xi_to_meanvarsqrt(self, xi_1, xi_2):
        return xi_1, xi_2

    def naturals_to_xi(self, nat_1, nat_2):
        return natural_to_meanvarsqrt(nat_1, nat_2)


#
# Auxiliary gaussian parameter conversion functions.
# 
# The following functions expect their first and second inputs to have shape
# DN1 and DNN, respectively. Return values are also of shapes DN1 and DNN.


def swap_dimensions(method):
    """
    Converts between GPflow indexing and tensorflow indexing
    `method` is a function that broadcasts over the first dimension (i.e. like all tensorflow matrix ops):
        `method` inputs DN1, DNN
        `method` outputs DN1, DNN
    :return: Function that broadcasts over the final dimension (i.e. compatible with GPflow):
        inputs: ND, DNN
        outputs: ND, DNN
    """
    @functools.wraps(method)
    def wrapper(a_nd, b_dnn, swap=True):
        if swap:
            if a_nd.get_shape().ndims != 2:  # pragma: no cover
                raise ValueError("The `a_nd` input must have 2 dimensions.")
            a_dn1 = tf.transpose(a_nd)[:, :, None]
            A_dn1, B_dnn = method(a_dn1, b_dnn)
            A_nd = tf.transpose(A_dn1[:, :, 0])
            return A_nd, B_dnn
        else:
            return method(a_nd, b_dnn)
    return wrapper

@swap_dimensions
def natural_to_meanvarsqrt(nat_1, nat_2):
    var_sqrt_inv = tf.cholesky(-2 * nat_2)
    var_sqrt = _inverse_lower_triangular(var_sqrt_inv)
    S = tf.matmul(var_sqrt, var_sqrt, transpose_a=True)
    mu = tf.matmul(S, nat_1)
    # We need the decomposition of S as L L^T, not as L^T L,
    # hence we need another cholesky.
    return mu, tf.cholesky(S)


@swap_dimensions
def meanvarsqrt_to_natural(mu, s_sqrt):
    s_sqrt_inv = _inverse_lower_triangular(s_sqrt)
    s_inv = tf.matmul(s_sqrt_inv, s_sqrt_inv, transpose_a=True)
    return tf.matmul(s_inv, mu), -0.5 * s_inv


@swap_dimensions
def natural_to_expectation(nat_1, nat_2):
    return meanvarsqrt_to_expectation(*natural_to_meanvarsqrt(nat_1, nat_2, swap=False), swap=False)


@swap_dimensions
def expectation_to_natural(eta_1, eta_2):
    return meanvarsqrt_to_natural(*expectation_to_meanvarsqrt(eta_1, eta_2, swap=False), swap=False)


@swap_dimensions
def expectation_to_meanvarsqrt(eta_1, eta_2):
    var = eta_2 - tf.matmul(eta_1, eta_1, transpose_b=True)
    return eta_1, tf.cholesky(var)


@swap_dimensions
def meanvarsqrt_to_expectation(m, v_sqrt):
    v = tf.matmul(v_sqrt, v_sqrt, transpose_b=True)
    return m, v + tf.matmul(m, m, transpose_b=True)

def _inverse_lower_triangular(M):
    """
    Take inverse of lower triangular (e.g. Cholesky) matrix. This function
    broadcasts over the first index.

    :param M: Tensor with lower triangular structure of shape DxNxN
    :return: The inverse of the Cholesky decomposition. Same shape as input.
    """
    if M.get_shape().ndims != 3:  # pragma: no cover
        raise ValueError("Number of dimensions for input is required to be 3.")
    D, N = tf.shape(M)[0], tf.shape(M)[1]
    I_DNN = tf.eye(N, dtype=M.dtype)[None, :, :] * tf.ones((D, 1, 1), dtype=M.dtype)
    return tf.matrix_triangular_solve(M, I_DNN)
