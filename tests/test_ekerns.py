# Copyright 2017 the GPflow authors.
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
# limitations under the License.from __future__ import print_function

import numpy as np
import tensorflow as tf
import gpflow

from gpflow.test_util import GPflowTestCase
from gpflow.test_util import session_context
from gpflow import kernels
from gpflow import ekernels

from numpy.testing import assert_allclose

import pytest


def _assert_pdeq(obj, a, b, k=None, i=-1, l=-1):
    obj.assertTrue(np.all(a.shape == b.shape))
    pdmax = np.max(np.abs(a / b - 1) * 100)
    # print("%s, %f" % (str(type(k)), pdmax))
    msg = "Percentage difference above threshold: {0}\nOn kernel: {1} ({2} / {3})"
    obj.assertTrue(pdmax < obj._threshold, msg=msg.format(pdmax, str(type(k)), i + 1, l))


def index_block(y, x, D):
    return np.s_[y * D:(y + 1) * D, x * D:(x + 1) * D]


class TriDiagonalBlockRep(object):
    """
    Transforms an unconstrained representation of a PSD block tri diagonal matrix to its PSD block representation.
    """

    def __init__(self):
        gpflow.transforms.Transform.__init__(self)

    def forward(self, x):
        """
        Transforms from the free state to the matrix of blocks.
        :param x: Unconstrained state (Nx2DxD), where D is the block size.
        :return: Return PSD blocks (2xNxDxD)
        """
        N, D = x.shape[0], x.shape[2]
        diagblocks = np.einsum('nij,nik->njk', x, x)
        ob = np.einsum('nij,nik->njk', x[:-1, :, :], x[1:, :, :])
        # ob = np.einsum('nij,njk->nik', x[:-1, :, :].transpose([0, 2, 1]), x[1:, :, :])
        offblocks = np.vstack((ob, np.zeros((1, D, D))))
        return np.array([diagblocks, offblocks])

    def forward_tensor(self, x):
        N, D = tf.shape(x)[0], tf.shape(x)[2]
        xm = tf.slice(x, [0, 0, 0], tf.stack([N - 1, -1, -1]))
        xp = x[1:, :, :]
        diagblocks = tf.matmul(x, x, transpose_a=True)
        offblocks = tf.concat_v2([tf.matmul(xm, xp, transpose_a=True), tf.zeros((1, D, D), 0, dtype=tf.float64)])
        return tf.stack([diagblocks, offblocks])

    def __str__(self):
        return "BlockTriDiagonal"


class TestKernExpDelta(GPflowTestCase):
    """
    Check whether the normal kernel matrix is recovered if a delta distribution is used. First initial test which should
    indicate whether things work or not.
    """

    @gpflow.defer_build()
    def setUp(self):
        self.test_graph = tf.Graph()
        with self.test_context():
            self.D = 2
            self.rng = np.random.RandomState(0)
            self.Xmu = self.rng.rand(10, self.D)
            self.Z = self.rng.rand(4, self.D)
            self.Xcov = np.zeros((self.Xmu.shape[0], self.D, self.D))
            self.Xcovc = np.zeros((self.Xmu.shape[0], self.D, self.D))
            k1 = ekernels.RBF(self.D, ARD=True)
            k1.lengthscales = self.rng.rand(2) + [0.5, 1.5]
            k1.variance = 0.3 + self.rng.rand()
            k2 = ekernels.RBF(self.D)
            k2.lengthscales = self.rng.rand(1) + [0.5]
            k2.variance = 0.3 + self.rng.rand()
            klin = ekernels.Linear(self.D, variance=0.3+self.rng.rand())
            self.kernels = [k1, klin, k2]

    def tearDown(self):
        GPflowTestCase.tearDown(self)
        for kernel in self.kernels:
            kernel.clear()

    def test_eKzxKxz(self):
        for k in self.kernels:
            with self.test_context():
                k.compile()
                psi2 = k.compute_eKzxKxz(self.Z, self.Xmu, self.Xcov)
                kernmat = k.compute_K(self.Z, self.Xmu)  # MxN
                kernouter = np.einsum('in,jn->nij', kernmat, kernmat)
                self.assertTrue(np.allclose(kernouter, psi2))

    def test_eKdiag(self):
        for k in self.kernels:
            with self.test_context():
                k.compile()
                kdiag = k.compute_eKdiag(self.Xmu, self.Xcov)
                orig = k.compute_Kdiag(self.Xmu)
                self.assertTrue(np.allclose(orig, kdiag))

    def test_exKxz_pairwise(self):
        covall = np.array([self.Xcov, self.Xcovc])
        for k in self.kernels:
            with self.test_context():
                if isinstance(k, ekernels.Linear):
                    continue
                k.compile()
                exKxz = k.compute_exKxz_pairwise(self.Z, self.Xmu, covall)
                Kxz = k.compute_K(self.Xmu[:-1, :], self.Z)  # NxM
                xKxz = np.einsum('nm,nd->nmd', Kxz, self.Xmu[1:, :])
                self.assertTrue(np.allclose(xKxz, exKxz))

#    def test_exKxz(self):
#        for k in self.kernels:
#            with self.test_session():
#                if isinstance(k, ekernels.Linear):
#                    continue
#                k.compile()
#                exKxz = k.compute_exKxz(self.Z, self.Xmu, self.Xcov)
#                Kxz = k.compute_K(self.Xmu, self.Z)  # NxM
#                xKxz = np.einsum('nm,nd->nmd', Kxz, self.Xmu)
#                self.assertTrue(np.allclose(xKxz, exKxz))

    def test_Kxz(self):
        for k in self.kernels:
            with self.test_context():
                k.compile()
                psi1 = k.compute_eKxz(self.Z, self.Xmu, self.Xcov)
                kernmat = k.compute_K(self.Z, self.Xmu)  # MxN
                self.assertTrue(np.allclose(kernmat, psi1.T))


class TestKernExpActiveDims(GPflowTestCase):
    _threshold = 0.5

    @gpflow.defer_build()
    def setUp(self):
        self.test_graph = tf.Graph()
        with self.test_context():
            self.N = 4
            self.D = 2
            self.rng = np.random.RandomState(0)
            self.Xmu = self.rng.rand(self.N, self.D)
            self.Z = self.rng.rand(3, self.D)
            unconstrained = self.rng.randn(self.N, 2 * self.D, self.D)
            t = TriDiagonalBlockRep()
            self.Xcov = t.forward(unconstrained)

            variance = 0.3 + self.rng.rand()

            k1 = ekernels.RBF(1, variance, active_dims=[0])
            k2 = ekernels.RBF(1, variance, active_dims=[1])
            klin = ekernels.Linear(1, variance, active_dims=[1])
            self.ekernels = [k1, k2, klin]  # Kernels doing the expectation in closed form, doing the slicing

            k1 = ekernels.RBF(1, variance)
            k2 = ekernels.RBF(1, variance)
            klin = ekernels.Linear(1, variance)
            self.pekernels = [k1, k2, klin]  # kernels doing the expectations in closed form, without slicing

            k1 = kernels.RBF(1, variance, active_dims=[0])
            klin = kernels.Linear(1, variance, active_dims=[1])
            self.kernels = [k1, klin]

            k1 = kernels.RBF(1, variance)
            klin = kernels.Linear(1, variance)
            self.pkernels = [k1, klin]

    def tearDown(self):
        GPflowTestCase.tearDown(self)
        for kernel in self.kernels + self.ekernels + self.pekernels + self.pkernels:
            kernel.clear()

    def test_quad_active_dims(self):
        with self.test_context():
            for k, pk in zip(self.kernels + self.ekernels, self.pkernels + self.pekernels):
                k.compile()
                pk.compile()
                a = k.compute_eKdiag(self.Xmu, self.Xcov[0, :, :, :])
                sliced = np.take(
                    np.take(self.Xcov, k.active_dims, axis=-1),
                    k.active_dims,
                    axis=-2)
                b = pk.compute_eKdiag(self.Xmu[:, k.active_dims], sliced[0, :, :, :])
                _assert_pdeq(self, a, b, k)

                a = k.compute_eKxz(self.Z, self.Xmu, self.Xcov[0, :, :, :])
                sliced = np.take(
                    np.take(self.Xcov, k.active_dims, axis=-1),
                    k.active_dims,
                    axis=-2)
                b = pk.compute_eKxz(
                    self.Z[:, k.active_dims],
                    self.Xmu[:, k.active_dims],
                    sliced[0, :, :, :])
                _assert_pdeq(self, a, b, k)

                a = k.compute_eKzxKxz(self.Z, self.Xmu, self.Xcov[0, :, :, :])
                sliced = np.take(
                    np.take(self.Xcov, k.active_dims, axis=-1),
                    k.active_dims,
                    axis=-2)
                b = pk.compute_eKzxKxz(self.Z[:, k.active_dims], self.Xmu[:, k.active_dims], sliced[0, :, :, :])
                _assert_pdeq(self, a, b, k)


class TestExpxKxzActiveDims(GPflowTestCase):
    _threshold = 0.5

    @gpflow.defer_build()
    def setUp(self):
        self.test_graph = tf.Graph()
        with self.test_context():
            self.rng = np.random.RandomState(0)

            self.N = 4
            self.D = 2
            self.Xmu = self.rng.rand(self.N, self.D)
            self.Z = self.rng.rand(3, self.D)
            unconstrained = self.rng.randn(self.N, 2 * self.D, self.D)
            t = TriDiagonalBlockRep()
            self.Xcov_pairwise = t.forward(unconstrained)
            self.Xcov = self.Xcov_pairwise[0]  # no cross-covariances

            variance = 0.3 + self.rng.rand()

            k1 = ekernels.RBF(1, variance, active_dims=[0])
            k2 = ekernels.RBF(1, variance, active_dims=[1])
            klin = ekernels.Linear(1, variance, active_dims=[1])
            self.ekernels = [k1, k2, klin]

            k1 = ekernels.RBF(2, variance)
            k2 = ekernels.RBF(2, variance)
            klin = ekernels.Linear(2, variance)
            self.pekernels = [k1, k2, klin]

            k1 = kernels.RBF(1, variance, active_dims=[0])
            klin = kernels.Linear(1, variance, active_dims=[1])
            self.kernels = [k1, klin]

            k1 = kernels.RBF(2, variance)
            klin = kernels.Linear(2, variance)
            self.pkernels = [k1, klin]

    def tearDown(self):
        super().tearDown()
        for kernel in self.kernels + self.ekernels + self.pekernels + self.pkernels:
            kernel.clear()

    def test_quad_active_dims_pairwise(self):
        for k, pk in zip(self.kernels, self.pkernels):
            with self.test_context():
                # TODO(@markvdw):
                # exKxz is interacts slightly oddly with `active_dims`.
                # It can't be implemented by simply dropping the dependence on certain inputs.
                # As we still need to output the outer product between x_{t-1} and K_{x_t, Z}.
                # So we can't do a comparison to a kernel that just takes a smaller X as an input.
                # It may be possible to do this though for a carefully crafted `Xcov`.
                # However, I'll leave that as a todo for now.

                k.input_size = self.Xmu.shape[1]
                pk.input_size = self.Xmu.shape[1]
                k.compile()
                pk.compile()
                a = k.compute_exKxz_pairwise(self.Z, self.Xmu, self.Xcov_pairwise)
                b = pk.compute_exKxz_pairwise(self.Z, self.Xmu, self.Xcov_pairwise)
                self.assertFalse(np.all(a == b))
                exp_shape = np.array([self.N - 1, self.Z.shape[0], self.D])
                msg = "Shapes incorrect:\n%s vs %s" % (str(a.shape), str(exp_shape))
                self.assertTrue(np.all(a.shape == exp_shape), msg=msg)
                k.clear()
                pk.clear()

        for k, pk in zip(self.ekernels, self.pekernels):
            with self.test_context():
                k.compile()
                pk.compile()
                with self.assertRaises(tf.errors.InvalidArgumentError):
                    k.compute_exKxz_pairwise(self.Z, self.Xmu, self.Xcov_pairwise)
                    pk.compute_exKxz_pairwise(self.Z, self.Xmu, self.Xcov_pairwise)

    def test_quad_active_dims(self):
        for k, pk in zip(self.kernels, self.pkernels):
            with self.test_context():
                # TODO(@markvdw):
                # exKxz is interacts slightly oddly with `active_dims`.
                # It can't be implemented by simply dropping the dependence on certain inputs.
                # As we still need to output the outer product between x_{t-1} and K_{x_t, Z}.
                # So we can't do a comparison to a kernel that just takes a smaller X as an input.
                # It may be possible to do this though for a carefully crafted `Xcov`.
                # However, I'll leave that as a todo for now.

                k.input_size = self.Xmu.shape[1]
                pk.input_size = self.Xmu.shape[1]
                k.compile()
                pk.compile()
                a = k.compute_exKxz(self.Z, self.Xmu, self.Xcov)
                b = pk.compute_exKxz(self.Z, self.Xmu, self.Xcov)
                self.assertFalse(np.all(a == b))
                exp_shape = np.array([self.N, self.Z.shape[0], self.D])
                msg = "Shapes incorrect:\n%s vs %s" % (str(a.shape), str(exp_shape))
                self.assertTrue(np.all(a.shape == exp_shape), msg=msg)
                k.clear()
                pk.clear()

        for k, pk in zip(self.ekernels, self.pekernels):
            with self.test_context():
                k.compile()
                pk.compile()
                with self.assertRaises(tf.errors.InvalidArgumentError):
                    k.compute_exKxz(self.Z, self.Xmu, self.Xcov)
                    pk.compute_exKxz(self.Z, self.Xmu, self.Xcov)


class TestKernProd(GPflowTestCase):
    """
    TestKernProd
    Need a separate test for this as Prod currently only supports diagonal Xcov matrices with non-overlapping kernels.
    """

    @gpflow.defer_build()
    def setUp(self):
        self.test_graph = tf.Graph()
        with self.test_context():
            self._threshold = 0.5
            self.rng = np.random.RandomState(0)
            self.N = 4
            self.D = 2

            # Test summed kernels, non-overlapping
            rbfvariance = 0.3 + self.rng.rand()
            rbfard = [self.rng.rand() + 0.5]
            linvariance = 0.3 + self.rng.rand()

            self.kernel = kernels.Prod([
                kernels.RBF(1, rbfvariance, rbfard, [1], False),
                kernels.Linear(1, linvariance, [0])
            ])

            self.ekernel = ekernels.Prod([
                ekernels.RBF(1, rbfvariance, rbfard, [1], False),
                ekernels.Linear(1, linvariance, [0])
            ])

            self.Xmu = self.rng.rand(self.N, self.D)
            self.Xcov = self.rng.rand(self.N, self.D)
            self.Z = self.rng.rand(2, self.D)

    def test_eKdiag(self):
        with self.test_context():
            self.kernel.compile()
            self.ekernel.compile()
            a = self.kernel.compute_eKdiag(self.Xmu, self.Xcov)
            b = self.ekernel.compute_eKdiag(self.Xmu, self.Xcov)
            _assert_pdeq(self, a, b)

    def test_eKxz(self):
        with self.test_context():
            self.kernel.compile()
            self.ekernel.compile()
            a = self.kernel.compute_eKxz(self.Z, self.Xmu, self.Xcov)
            b = self.ekernel.compute_eKxz(self.Z, self.Xmu, self.Xcov)
            _assert_pdeq(self, a, b)

    def test_eKzxKxz(self):
        with self.test_context():
            self.kernel.compile()
            self.ekernel.compile()
            a = self.kernel.compute_eKzxKxz(self.Z, self.Xmu, self.Xcov)
            b = self.ekernel.compute_eKzxKxz(self.Z, self.Xmu, self.Xcov)
            _assert_pdeq(self, a, b)


class DataExp:
    threshold = 1e-6
    rng = np.random.RandomState(0)
    N = 4
    D = 2
    Xmu = rng.rand(N, D)
    Z = rng.rand(2, D)

    Xcov_diag = 0.05 + rng.rand(N, D)
    Xcov = np.zeros((Xcov_diag.shape[0], Xcov_diag.shape[1], Xcov_diag.shape[1]))
    Xcov[(np.s_[:],) + np.diag_indices(Xcov_diag.shape[1])] = Xcov_diag

    add_args = (0.3 + rng.rand(), [rng.rand() + 0.5], 0.3 + rng.rand())

    linear_args = (D, 0.3 + rng.rand(), None)
    rbf_args = (D, 0.3 + rng.rand(),
                rng.rand(2) + [0.5, 1.5],
                None, True)

    @classmethod
    def create_rbf_kernel(cls, module):
        return module.RBF(*cls.rbf_args)

    @classmethod
    def create_linear_kernel(cls, module):
        return module.Linear(*cls.linear_args)

    @classmethod
    def create_add_kernel1(cls, module):
        variance, ard, linvariance = cls.add_args
        kern = module.Add([
            module.RBF(1, variance, ard, [1], False),
            module.Linear(1, linvariance, [0])
        ])

        kern.input_size = kern.input_dim
        for k in kern.kern_list:
            k.input_size = kern.input_size

        return kern

    @classmethod
    def create_add_kernel2(cls, module):
        variance, ard, linvariance = cls.add_args
        return module.Add([
            module.RBF(cls.D, variance, ard),
            module.Linear(cls.D, linvariance)
        ])


kernel_modules = [kernels, ekernels]
create_exp_diag_kernels = [
    DataExp.create_rbf_kernel,
    DataExp.create_linear_kernel,
    DataExp.create_add_kernel1,
    DataExp.create_add_kernel2,
]

@pytest.mark.parametrize('module', kernel_modules)
@pytest.mark.parametrize('create_kernel', create_exp_diag_kernels)
@session_context()
def test_exp_diag_eKdiag(module, create_kernel):
    k = create_kernel(module)
    a = k.compute_eKdiag(DataExp.Xmu, DataExp.Xcov)
    b = k.compute_eKdiag(DataExp.Xmu, DataExp.Xcov_diag)
    assert_allclose(a, b, rtol=DataExp.threshold)


@pytest.mark.parametrize('module', kernel_modules)
@pytest.mark.parametrize('create_kernel', create_exp_diag_kernels)
@session_context()
def test_exp_diag_eKxz(module, create_kernel):
    k = create_kernel(module)
    a = k.compute_eKxz(DataExp.Z, DataExp.Xmu, DataExp.Xcov)
    b = k.compute_eKxz(DataExp.Z, DataExp.Xmu, DataExp.Xcov_diag)
    assert_allclose(a, b, rtol=DataExp.threshold)


@pytest.mark.parametrize('module', kernel_modules)
@pytest.mark.parametrize('create_kernel', create_exp_diag_kernels)
@session_context()
def test_exp_diag_eKzxKxz(module, create_kernel):
    k = create_kernel(module)
    a = k.compute_eKzxKxz(DataExp.Z, DataExp.Xmu, DataExp.Xcov)
    b = k.compute_eKzxKxz(DataExp.Z, DataExp.Xmu, DataExp.Xcov_diag)
    assert_allclose(a, b, rtol=DataExp.threshold)



class DataExpQuadrature(DataExp):
    threshold = 1e-3
    num_gauss_hermite_points = 50  # more may be needed to reach tighter tolerances, try 100.
    unconstrained = DataExp.rng.randn(DataExp.N, 2 * DataExp.D, DataExp.D)
    Xcov = TriDiagonalBlockRep().forward(unconstrained)


create_exp_quadrature_kernels = [
    DataExpQuadrature.create_rbf_kernel,
    DataExpQuadrature.create_linear_kernel,
    DataExpQuadrature.create_add_kernel1,
    DataExpQuadrature.create_add_kernel2,
]


@pytest.mark.parametrize('create_kernel', create_exp_quadrature_kernels)
def test_exp_quadrature_diag(create_kernel):
    Xmu = DataExpQuadrature.Xmu
    Xcov = DataExpQuadrature.Xcov[0, :, :, :]
    with session_context():
        k = create_kernel(kernels)
        a = k.compute_eKdiag(Xmu, Xcov)
    with session_context():
        ek = create_kernel(ekernels)
        b = ek.compute_eKdiag(Xmu, Xcov)
    assert_allclose(a, b, rtol=DataExpQuadrature.threshold)


@pytest.mark.parametrize('create_kernel', create_exp_quadrature_kernels)
@session_context()
def test_exp_quadrature_eKxz(create_kernel):
    Z = DataExpQuadrature.Z
    Xmu = DataExpQuadrature.Xmu
    Xcov = DataExpQuadrature.Xcov[0, :, :, :]
    with session_context():
        k = create_kernel(kernels)
        k.num_gauss_hermite_points = DataExpQuadrature.num_gauss_hermite_points
        a = k.compute_eKxz(Z, Xmu, Xcov)
    with session_context():
        ek = create_kernel(ekernels)
        b = ek.compute_eKxz(Z, Xmu, Xcov)
    assert_allclose(a, b, rtol=DataExpQuadrature.threshold)


@pytest.mark.parametrize('create_kernel', create_exp_quadrature_kernels)
def test_exp_quadrature_eKzxKxz(create_kernel):
    Z = DataExpQuadrature.Z
    Xmu = DataExpQuadrature.Xmu
    Xcov = DataExpQuadrature.Xcov[0, :, :, :]
    with session_context():
        k = create_kernel(kernels)
        k.num_gauss_hermite_points = DataExpQuadrature.num_gauss_hermite_points
        a = k.compute_eKzxKxz(Z, Xmu, Xcov)
    with session_context():
        ek = create_kernel(ekernels)
        b = ek.compute_eKzxKxz(Z, Xmu, Xcov)
    assert_allclose(a, b, rtol=DataExpQuadrature.threshold)


# xKxz does not work with slicing yet
@pytest.mark.parametrize('create_kernel', create_exp_quadrature_kernels[:2])
def test_exp_quadrature_exKxz_pairwise(create_kernel):
    Z = DataExpQuadrature.Z
    Xmu = DataExpQuadrature.Xmu
    Xcov = DataExpQuadrature.Xcov
    with session_context():
        k = create_kernel(kernels)
        k.num_gauss_hermite_points = DataExpQuadrature.num_gauss_hermite_points
        a = k.compute_exKxz_pairwise(Z, Xmu, Xcov)
    with session_context():
        ek = create_kernel(ekernels)
        b = ek.compute_exKxz_pairwise(Z, Xmu, Xcov)
    assert_allclose(a, b, rtol=DataExpQuadrature.threshold)


# xKxz does not work with slicing yet
@pytest.mark.parametrize('create_kernel', create_exp_quadrature_kernels[:2])
def test_exp_quadrature_exKxz(create_kernel):
    Z = DataExpQuadrature.Z
    Xmu = DataExpQuadrature.Xmu
    Xcov = DataExpQuadrature.Xcov[0]
    with session_context():
        k = create_kernel(kernels)
        k.num_gauss_hermite_points = DataExpQuadrature.num_gauss_hermite_points
        a = k.compute_exKxz(Z, Xmu, Xcov)
    with session_context():
        ek = create_kernel(ekernels)
        b = ek.compute_exKxz(Z, Xmu, Xcov)
    assert_allclose(a, b, rtol=DataExpQuadrature.threshold)


@pytest.mark.parametrize('create_kernel', [create_exp_quadrature_kernels[0]])
@session_context()
def test_exp_quadrature_switch(create_kernel):
    k = create_kernel(kernels)
    k.num_gauss_hermite_points = 0
    Z = DataExpQuadrature.Z
    Xmu = DataExpQuadrature.Xmu
    Xcov = DataExpQuadrature.Xcov[0, :, :, :]
    with pytest.raises(RuntimeError):
        k.compute_eKzxKxz(Z, Xmu, Xcov)


class TestAddCrossCalcs(GPflowTestCase):
    _threshold = 0.5

    @gpflow.defer_build()
    def setUp(self):
        self.test_graph = tf.Graph()
        self.rng = np.random.RandomState(0)
        self.N = 4
        self.D = 2

        self.rbf = ekernels.RBF(self.D, ARD=True)
        self.rbf.lengthscales = self.rng.rand(2) + [0.5, 1.5]
        self.rbf.variance = 0.3 + self.rng.rand()
        self.lin = ekernels.Linear(self.D)
        self.lin.variance = 0.3 + self.rng.rand()
        self.add = ekernels.Add([self.rbf, self.lin])

        self.Xmu = self.rng.rand(self.N, self.D)
        self.Z = self.rng.rand(2, self.D)
        unconstrained = self.rng.randn(self.N, 2 * self.D, self.D)
        t = TriDiagonalBlockRep()
        self.Xcov = t.forward(unconstrained)[0, :, :, :]

    def tearDown(self):
        GPflowTestCase.tearDown(self)
        self.add.clear()

    def test_cross_quad(self):
        with self.test_context() as session:
            self.add.num_gauss_hermite_points = 50
            self.add.compile()
            tfZ = tf.placeholder(tf.float64)
            tfXmu = tf.placeholder(tf.float64)
            tfXcov = tf.placeholder(tf.float64)
            tfa = self.add.Linear_RBF_eKxzKzx(self.add.kern_list[0], self.add.kern_list[1], tfZ, tfXmu, tfXcov)
            tfb = self.add.quad_eKzx1Kxz2(self.add.kern_list[0], self.add.kern_list[1], tfZ, tfXmu, tfXcov)
            feed_dict = {tfZ: self.Z,
                         tfXmu: self.Xmu,
                         tfXcov: self.Xcov}
            a, b = session.run((tfa, tfb), feed_dict=feed_dict)
            _assert_pdeq(self, a, b)


if __name__ == "__main__":
    tf.test.main()
