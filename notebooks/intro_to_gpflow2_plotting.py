import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def summary_matplotlib_image(figures, step, fmt="png"):
    for name, fig in figures.items():
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, bbox_inches='tight')
        buf.seek(0)
        image = buf.getvalue()
        image = tf.image.decode_image(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        tf.summary.image(name=name, data=image, step=step)


def plotting_regression(X, Y, xx, mean, var, samples):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.plot(xx, mean, 'C0', lw=2)
    ax.fill_between(xx[:, 0],
                    mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
                    mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
                    color='C0',
                    alpha=0.2)
    ax.plot(X, Y, 'kx')
    ax.plot(xx, samples[:, :, 0].numpy().T, 'C0', linewidth=.5)
    ax.set_ylim(-2., +2.)
    ax.set_xlim(0, 10)
    plt.close()
    return fig
