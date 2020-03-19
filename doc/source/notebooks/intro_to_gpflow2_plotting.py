import numpy as np


def plotting_regression(X, Y, xx, mean, var, samples, *, fig=None, ax=None):
    ax.plot(xx, mean, "C0", lw=2)
    ax.fill_between(
        xx[:, 0],
        mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
        mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
        color="C0",
        alpha=0.2,
    )
    ax.plot(X, Y, "kx")
    ax.plot(xx, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
    ax.set_ylim(-2.0, +2.0)
    ax.set_xlim(0, 10)
