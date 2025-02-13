import numpy as np
import tensorflow as tf
import psutil

import gpflow

X = np.array(
    [
        [0.865], [0.666], [0.804], [0.771], [0.147], [0.866], [0.007], [0.026],
        [0.171], [0.889], [0.243], [0.028],
    ]
)
Y = np.array(
    [
        [1.57], [3.48], [3.12], [3.91], [3.07], [1.35], [3.80], [3.82], [3.49],
        [1.30], [4.00], [3.82],
    ]
)

model = gpflow.models.GPR((X, Y), kernel=gpflow.kernels.SquaredExponential())
opt = gpflow.optimizers.Scipy()

print("[0]", psutil.virtual_memory())

for i in range(400):
    training_loss = model.training_loss_closure(compile=True)
    opt.minimize(training_loss, model.trainable_variables, compile=True)
    print(f"[{i+1}]", psutil.virtual_memory())

