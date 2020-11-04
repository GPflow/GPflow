import tensorflow as tf
import numpy as np
from .base import Kernel


class Orbit:
    @property
    def size(self):
        raise NotImplementedError

    def get_orbit(self, X):
        raise NotImplementedError

    def __call__(self, X):
        return tf.stack(self.get_orbit(X))


class Nop(Orbit):
    """ Null element """

    size = 1

    def get_orbit(self, X):
        return [X]


class Permute2D(Orbit):
    """ 2D permutation symmetry: (x1,x2) <-> (x2,x1) """

    size = 2

    def get_orbit(self, X):
        assert X.shape[-1] == 2
        return [X, X[..., ::-1]]


class DiscreteRotation(Orbit):
    """ n-fold rotational symmetry """

    def __init__(self, n: int):
        self.n = n

    @property
    def size(self):
        return self.n

    def _rotation_matrix(self):
        angle = 2 * np.pi / self.n
        ca = np.cos(angle)
        sa = np.sin(angle)
        return np.array([[ca, sa], [-sa, ca]])

    def get_orbit(self, X):
        R = self._rotation_matrix()
        orbit = [X]
        for _ in range(self.n - 1):
            X = X @ R
            orbit.append(X)
        assert len(orbit) == self.size
        return orbit


class Compose(Orbit):
    """ Composition of orbits """

    def __init__(self, orbits):
        self._orbits = orbits

    @property
    def size(self):
        return np.prod([o.size for o in self._orbits])

    def __call__(self, X):
        assert len(self._orbits) == 2, "hardcoded for two orbits"
        XO1 = self._orbits[0](X)
        dim = tf.shape(XO1)[-1]
        XO2 = self._orbits[1](tf.reshape(XO1, [-1, dim]))
        return tf.reshape(XO2, [self.size, -1, dim])


class InvariantKernel(Kernel):
    """ Assumes a finite orbit """

    def __init__(self, base: Kernel, orbit: Orbit):
        super().__init__()
        self.base = base
        self.orbit = orbit

    def K(self, X, X2=None):
        N = tf.shape(X)[0]
        Osize = self.orbit.size
        dim = tf.shape(X)[-1]
        XO = tf.reshape(self.orbit(X), [-1, dim])
        XO2 = tf.reshape(self.orbit(X2), [-1, dim]) if X2 is not None else None
        base_k = tf.reshape(self.base.K(XO, XO2), [Osize, N, Osize, N])
        return tf.reduce_sum(base_k, [0, 2])

    def K_diag(self, X):
        XO = self.orbit(X)
        basek = self.base.K_diag(XO)
        return tf.reduce_sum(basek, axis=0)


class SnowflakeKernel(InvariantKernel):
    def __init__(self, base: Kernel):
        snowflake_orbit = Compose([Permute2D(), DiscreteRotation(6)])
        super().__init__(base, snowflake_orbit)
