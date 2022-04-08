import jax.numpy as np
import numpy.testing as npt
from gpflow.base import AnyNDArray
from jax import grad, jit, random

from jn_gps.clastion import Clastion, derived, put
from jn_gps.clastion.integration.check_shapes import shape
from jn_gps.clastion.integration.jax import arrayput
from jn_gps.clastion.utilities import multi_set, root, to_loss_function


def test_jax_integration() -> None:
    class LinearModel(Clastion):

        weights = arrayput(shape("[n_features]"))
        offset = arrayput(shape("[]"))
        features = arrayput(shape("[batch..., n_features]"))

        @derived(shape("[batch...]"))
        def prediction(self) -> AnyNDArray:
            return np.einsum("...i,i -> ...", self.features, self.weights) + self.offset

    class Loss(Clastion):

        model = put(LinearModel)
        x = arrayput(shape("[batch..., n_features]"))
        y = arrayput(shape("[batch...]"))

        @derived(shape("[batch...]"))
        def prediction(self) -> AnyNDArray:
            return self.model(features=self.x).prediction

        @derived(shape("[]"))
        def loss(self) -> AnyNDArray:
            return np.mean((self.y - self.prediction) ** 2)

    target_weights = np.array([0.3, 0.6])
    target_offset = np.array(0.5)

    key = random.PRNGKey(20220506)
    x = random.normal(key, (100, 2))
    y = LinearModel(weights=target_weights, offset=target_offset, features=x).prediction
    loss = Loss(model=LinearModel(weights=np.zeros((2,)), offset=np.zeros(())), x=x, y=y)

    loss_fn, params = to_loss_function(loss, [root.model.weights, root.model.offset], root.loss)
    loss_grad = jit(grad(loss_fn))

    for _ in range(100):
        param_grads = loss_grad(params)
        params = {k: v - 0.1 * param_grads[k] for k, v in params.items()}

    loss = multi_set(loss, params)
    npt.assert_allclose(target_weights, loss.model.weights, atol=1e-7)  # type: ignore
    npt.assert_allclose(target_offset, loss.model.offset, atol=1e-7)  # type: ignore
