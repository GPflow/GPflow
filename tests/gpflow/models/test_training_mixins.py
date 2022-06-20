import numpy as np
import tensorflow as tf

import gpflow
from gpflow.base import RegressionData


class DummyModel(gpflow.models.BayesianModel, gpflow.models.ExternalDataTrainingLossMixin):
    # type-ignore is because of changed method signature:
    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:  # type: ignore[override]
        X, Y = data
        return tf.reduce_sum(X * Y)


def test_training_loss_closure_with_minibatch() -> None:
    N = 13
    B = 5
    num_batches = int(np.ceil(N / B))

    data = (np.random.randn(N, 2), np.random.randn(N, 1))
    dataset = tf.data.Dataset.from_tensor_slices(data)

    model = DummyModel()

    training_loss_full_data = model.training_loss_closure(data, compile=True)
    loss_full = training_loss_full_data()

    it = iter(dataset.batch(B))
    training_loss_minibatch = model.training_loss_closure(it, compile=True)
    batch_losses = [training_loss_minibatch() for _ in range(num_batches)]

    np.testing.assert_allclose(loss_full, np.sum(batch_losses))
