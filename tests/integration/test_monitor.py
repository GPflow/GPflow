import pprint

import numpy as np
import tensorflow as tf

import gpflow
from gpflow.utilities import parameter_dict, print_summary
from gpflow.utilities.monitor import ModelToTensorBoardTask


if __name__ == "__main__":
    print("Hello world")
    X = np.random.randn(100, 2)  # [N, 2]
    Y = np.sin(np.sum(X, axis=-1, keepdims=True))  # [N, 1]
    kernel = (
        gpflow.kernels.SquaredExponential(lengthscale=[1.0, 2.0])
        + gpflow.kernels.Linear()
    )
    m = gpflow.models.GPR((X, Y), kernel, noise_variance=0.01)
    print_summary(m)

    params = parameter_dict(m)
    pprint.pprint(params)

    @tf.function
    def closure():
        return -m.log_likelihood()

    opt = tf.optimizers.Adam()

    task = ModelToTensorBoardTask(m, "logs", 1)

    STEPS = 100
    for step in range(STEPS):
        print(step)
        opt.minimize(closure, m.trainable_variables)
        task(step)
