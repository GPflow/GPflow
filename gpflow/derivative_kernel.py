# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews
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


from __future__ import print_function, absolute_import
from functools import reduce
import itertools
import warnings

import tensorflow as tf
import numpy as np
from .param import Param, Parameterized, AutoFlow
from . import transforms
from ._settings import settings
from .quadrature import hermgauss, mvhermgauss, mvnquad
from . import kernels


float_type = settings.dtypes.float_type
int_type = settings.dtypes.int_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64



class DifferentialObservationsKernelDynamic(kernels.Kern):
    """
    Differential kernels
    These are kernels between observations of the function and
    observation of function derivatives. see eg:
    http://mlg.eng.cam.ac.uk/pub/pdf/SolMurLeietal03.pdf
    Solak, Ercan, et al. "Derivative observations in Gaussian process models of dynamic systems.
    " Advances in neural information processing systems. 2003.

    This particular kernel can work with dynamically defined observations. In other words one
    can define after building the graph the observations. Because TensorFlow has to have a static
    graph this class has to build the possible combinations of derivatives at compile time.
    A switch is then used to pick the correct portion of the graph to evaluate at run time.
    We have only defined the kernel to deal with upto second order derivatives.

    When feeding the data to the GP one must define an extra dimension. This defines
    the number of derivatives in this corresponding dimension direction the corresponding
    observation records.

    For instance the x tensor:
    x = [[a,b,0,1],
         [c,d,0,0],
         [e,f,2,0]]

    would mean you have seen three observations at [a,b], [c,d] and [e,f].
    and these observations will be respectively
    df/dx_2|x=[a,b]
    f|x=[c,d]
    and d^2f/(dx_1^2)|x=[e,f].
    (you have to pass into this class the variable obs_dims, which denotes the number of dimensions
    of the observations, in this case 2).

    Undefined behaviour if following condiitions not met:
        * gradient denotation should be positive integers
        * only can do up to second derivatives
    """

    def __init__(self, input_dim, base_kernel, obs_dims, active_dims=None):
        kernels.Kern.__init__(self, input_dim, active_dims)
        self.obs_dims = obs_dims
        self.base_kernel = base_kernel


    def K(self, X, X2=None):
        # Split X up into two separate vectors (do this as when we do tf.gradients
        # we only actually want to differentiate
        if X2 is None:
            X2 = tf.identity(X)
            X = tf.identity(X)

        x1, d1 = self._split_x_into_locs_and_grad_information(X)
        x2, d2 = self._split_x_into_locs_and_grad_information(X2)

        d1 = self._convert_grad_info_into_indics(d1)
        d2 = self._convert_grad_info_into_indics(d2)


        # Compute the kernel assuming no gradient observations
        raw_kernel = self.base_kernel.K(x1, x2)

        new_k = self._k_correct_dynamic(raw_kernel, x1, x2, d1, d2)
        return new_k

    def Kdiag(self, X):
        k = self.K(X)
        # So we have to solve the diag going through the whole K matrix, as some of the Kdiag
        # implementations make simplifications which means that the gradients will not be correct.
        # For instance the stationary kernel returns just the variances so as X is ignored
        # it will not differentiate properly
        return tf.diag_part(k)

    def _split_x_into_locs_and_grad_information(self, x):
        locs = x[:, :self.obs_dims]
        grad_info = x[:, -self.obs_dims:]
        return locs, grad_info

    def _convert_grad_info_into_indics(self, grad_info_matrix):
        """
        This function takes gradient information in the form given to the class -- ie an
        integer mask telling how many times the gradient has been taken in that direction and
        converts it to the derivative information matrix form.
        An example derivative information matrix
        [[2, 0, 1],
         [1, 1, -1],
         [0, -1, -1]]

        would mean the observations corresponding to these data points are:
        [ d^2f_1/(dx_a1 dx_a2), df_2/dx_b2, f_3].

        We assume that the maximum number of derivatives will be two but do not check this so
        undefined behaviour if you have given more than two.
        """
        deriv_info_matrix = tf.to_int32(grad_info_matrix)
        number_grads = tf.reduce_sum(deriv_info_matrix, axis=1)

        first_index = tf.argmax(deriv_info_matrix, axis=1, output_type=tf.int32)

        # Having worked out where the first derivative is taken from we
        #  now remove it from the records.
        remaining = deriv_info_matrix - tf.one_hot(first_index, depth=tf.shape(deriv_info_matrix)[1],
                                                   dtype=tf.int32)

        second_index = tf.argmax(remaining, axis=1, output_type=tf.int32)

        deriv_info_matrix = tf.transpose(tf.stack((number_grads, first_index, second_index), axis=0))
        return deriv_info_matrix

    def _k_correct_dynamic(self, k, xl, xr, deriv_info_left, deriv_info_right):
        k_shape = tf.shape(k)
        k_orig = k

        indcs_x1 = tf.range(0, tf.shape(xl)[0])[:, None] + tf.zeros(tf.shape(k), dtype=tf.int32)
        indcs_x2 = tf.range(0, tf.shape(xr)[0])[None, :] + tf.zeros(tf.shape(k), dtype=tf.int32)

        elems = [tf.reshape(t, (-1,)) for t in (indcs_x1, indcs_x2)]

        def calc_derivs(tensor_in):
            idxl = tensor_in[0]
            idxr = tensor_in[1]

            k = k_orig[idxl, idxr]

            idx_i = deriv_info_left[idxl, 1]
            idx_j = deriv_info_left[idxl, 2]
            idx_k = deriv_info_right[idxr, 1]
            idx_m = deriv_info_right[idxr, 2]

            # First order derivatives
            dk__dxli = lambda: tf.gradients(k, xl)[0][idxl, idx_i]
            dk__dxrk = lambda: tf.gradients(k, xr)[0][idxr, idx_k]

            # Second order derivatives
            dk__dxlj_dxli_ = tf.gradients(dk__dxli(), xl)[0][idxl, idx_j]
            dk__dxli_dxrk_ = tf.gradients(dk__dxrk(), xl)[0][idxl, idx_i]
            dk__dxrm_dxrk_ = tf.gradients(dk__dxrk(), xr)[0][idxr, idx_m]
            dk__dxlj_dxli = lambda: dk__dxlj_dxli_
            dk__dxli_dxrk = lambda: dk__dxli_dxrk_
            dk__dxrm_dxrk = lambda: dk__dxrm_dxrk_

            # Third order derivatives
            dk__dxlj_dxli_dxrk = lambda: tf.gradients(dk__dxli_dxrk_, xl)[0][idxl, idx_j]
            dk__dxli_dxrm_dxrk = lambda: tf.gradients(dk__dxrm_dxrk_, xl)[0][idxl, idx_i]

            # Fourth order derivatives
            dk__dxlj_dxli_dxrm_dxrk = lambda: tf.gradients(dk__dxli_dxrm_dxrk(), xl)[0][idxl, idx_j]

            num_left_derivs = deriv_info_left[idxl, 0]
            num_right_derivs = deriv_info_right[idxr, 0]
            k_new = tf.case(
                [
                    # Zeroth order
                    # ... is done by default
                    # First order
                    (tf.logical_and(tf.equal(num_left_derivs, 1), tf.equal(num_right_derivs, 0)),
                     dk__dxli),
                    (tf.logical_and(tf.equal(num_left_derivs, 0), tf.equal(num_right_derivs, 1)),
                     dk__dxrk),
                    # Second order
                    (tf.logical_and(tf.equal(num_left_derivs, 2), tf.equal(num_right_derivs, 0)),
                     dk__dxlj_dxli),
                    (tf.logical_and(tf.equal(num_left_derivs, 1), tf.equal(num_right_derivs, 1)),
                     dk__dxli_dxrk),
                    (tf.logical_and(tf.equal(num_left_derivs, 0), tf.equal(num_right_derivs, 2)),
                     dk__dxrm_dxrk),
                    # Third order
                    (tf.logical_and(tf.equal(num_left_derivs, 2), tf.equal(num_right_derivs, 1)),
                     dk__dxlj_dxli_dxrk),
                    (tf.logical_and(tf.equal(num_left_derivs, 1), tf.equal(num_right_derivs, 2)),
                     dk__dxli_dxrm_dxrk),
                    # Fourth order
                    (tf.logical_and(tf.equal(num_left_derivs, 2), tf.equal(num_right_derivs, 2)),
                     dk__dxlj_dxli_dxrm_dxrk),
                ], default=lambda: k, exclusive=True
            )

            return k_new

        new_kernel = tf.map_fn(calc_derivs, elems, dtype=tf.float64)
        new_kernel_reshaped = tf.reshape(new_kernel, k_shape)
        return new_kernel_reshaped



class RBFDerivativeKern(DifferentialObservationsKernelDynamic):
    """
    Have the analytical expressions for the RBF kernel. We hope that this allows for faster
    evaluation.
    """
    def __init__(self, input_dim, obs_dims, active_dims=None):
        base_kernel = kernels.RBF(input_dim, active_dims=active_dims)
        DifferentialObservationsKernelDynamic.__init__(self, input_dim, base_kernel,
                                                       obs_dims, active_dims=active_dims)


    def _k_correct_dynamic(self, orig_kern, xl, xr, deriv_info_left, deriv_info_right):
        """
        So for speed reasons we want this to be as vectorised as possible. Also due to static graph
        restrictions we will define all the possible routes through. A dynamic stitch will be used
        at the end to choose the right version.
        See base class for indexes.
        """
        # Note there are different ways to do this. One may try and do dynamic partition, do the
        # respective derivative calculations and then stitch everything back together at the end
        # using dynamic stitch. Instead we have decided to opt for calculating everything everywhere
        # and then selecting teh correct result via gather_nd at the end. Done this as some of
        # the intermediate calculations actually come up again later on and so may gain speed by
        # computing them once in a large batch. We also thought that it probably would be slightly
        # easier to code. If this method is slow we could consider the dynamic partition and stitch
        # equivalent instead

        xl_min_xr_for_idx_i = self._compute_pairwise_distances(xl, xr, deriv_info_left[:, 1], left=True)
        xl_min_xr_for_idx_j = self._compute_pairwise_distances(xl, xr, deriv_info_left[:, 2], left=True)
        xl_min_xr_for_idx_k = self._compute_pairwise_distances(xl, xr, deriv_info_right[:, 1], left=False)
        xl_min_xr_for_idx_m = self._compute_pairwise_distances(xl, xr, deriv_info_right[:, 2], left=False)

        ell_sq_i = tf.gather(self.base_kernel.lengthscales, deriv_info_left[:, 1])[:, None]**(-2)
        ell_sq_j = tf.gather(self.base_kernel.lengthscales, deriv_info_left[:, 2])[:, None]**(-2)
        ell_sq_k = tf.gather(self.base_kernel.lengthscales, deriv_info_right[:, 1])[None, :]**(-2)
        ell_sq_m = tf.gather(self.base_kernel.lengthscales, deriv_info_right[:, 2])[None, :]**(-2)


        # First derivatives
        dk__dxli = - orig_kern * xl_min_xr_for_idx_i * ell_sq_i
        dk__dxrk = orig_kern * xl_min_xr_for_idx_k * ell_sq_k


        # Second derivatives
        where_i_equals_j = tf.cast(tf.equal(deriv_info_left[:, 1], deriv_info_left[:, 2])[:, None], float_type)
        dk__dxlj_dxli = -where_i_equals_j * ell_sq_j * orig_kern + -ell_sq_j* xl_min_xr_for_idx_j * dk__dxli
        where_i_equals_k = tf.cast(tf.equal(deriv_info_left[:, 1][:, None], deriv_info_right[:, 1][None, :]), float_type)
        dk__dxli_dxrk = where_i_equals_k * ell_sq_i * orig_kern - ell_sq_i * xl_min_xr_for_idx_i * dk__dxrk
        where_m_equals_k = tf.cast(tf.equal(deriv_info_right[:, 1], deriv_info_right[:, 2]), float_type)[None, :]
        dk__dxrm_dxrk = -where_m_equals_k * ell_sq_m * orig_kern + ell_sq_m * xl_min_xr_for_idx_m * dk__dxrk


        stacked = tf.stack([orig_kern, dk__dxli, dk__dxrk, dk__dxli_dxrk])
        reshaped = tf.reshape(stacked, [tf.shape(stacked)[0], -1])

        choice = tf.reshape(deriv_info_left[:, 0, None] + 2*tf.expand_dims(deriv_info_right[:, 0], axis=0),  (-1,))
        arange = tf.range(tf.size(choice))
        coords = tf.stack((choice, arange), axis=1)

        new_k = tf.gather_nd(reshaped, coords)
        new_k_correct_shape = tf.reshape(new_k, tf.shape(orig_kern))
        return new_k_correct_shape




    def _compute_pairwise_distances(self, x1, x2, idx, left=True):
        """
        :param idx:
        :param left:  if true idx refers to the left item but if not then referes to righ
        """


        num_left = tf.shape(x1)[0]
        num_right = tf.shape(x2)[0]

        if left:
            ra = tf.range(num_left)
            idx_all = tf.stack((ra, idx), axis=1)
            x1s = tf.gather_nd(x1, idx_all)
            x1_full = tf.expand_dims(x1s, axis=1)

            rar = tf.range(num_right)
            rar_rep = tf.tile(rar, [num_left])
            idxr = tensorflow_repeats(idx, num_right)
            idx_all_r = tf.stack((rar_rep, idxr), axis=1)
            x2_full_flat = tf.gather_nd(x2, idx_all_r)
            x2_full = tf.reshape(x2_full_flat, (num_left, num_right))
        else:
            ra = tf.range(num_right)
            idx_all = tf.stack((ra, idx), axis=1)
            x2s = tf.gather_nd(x2, idx_all)
            x2_full = tf.expand_dims(x2s, axis=0)

            ral = tf.range(num_left)
            ral_rep = tensorflow_repeats(ral, num_right)
            idxl = tf.tile(idx, [num_left])
            idx_all_l = tf.stack((ral_rep, idxl), axis=1)
            x1_full_flat = tf.gather_nd(x1, idx_all_l)
            x1_full = tf.reshape(x1_full_flat, (num_left, num_right))

        diff = x1_full - x2_full

        # ra = tf.range(tf.shape(x1)[0])
        # idx_all = tf.stack((ra, idx), axis=1)
        # x1s = tf.gather_nd(x1, idx_all)
        # x2s = tf.gather_nd(x2, idx_all)
        # x1s[:, None] - x2s[None, :]
        return diff


def tensorflow_repeats(vec, num_repeats):
    """
    This is TensorFlow version of numpy repeats.
    # for instance to repeat vec=[0,1,2] 3 times we want:
    # [0,0,0,1,1,1,2,2,2]
    :param vec: vec should be one dimensional. we'll say it has length N
    :param num_repeats: a scalar saying how many time to repeat. we'll call this R
    """

    vec = tf.expand_dims(vec, axis=1)  # it is now a column vector (Nx1)
    vec = tf.tile(vec, [1, num_repeats])  # it is now a matrix (NxR)
    vec = tf.reshape(vec, (-1,))  # it is now the NR vector of the form we want.
    return vec