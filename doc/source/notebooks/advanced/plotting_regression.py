import numpy as np
import matplotlib.pyplot as plt


def plotting_regression(X, Y, xx, mean, var, samples):
    fig = plt.figure(0)
    ## plot
    plt.figure(figsize=(12, 6))
    plt.plot(xx, mean, 'C0', lw=2)
    plt.fill_between(xx[:,0],
                     mean[:,0] - 1.96 * np.sqrt(var[:,0]),
                     mean[:,0] + 1.96 * np.sqrt(var[:,0]),
                     color='C0', alpha=0.2)
    plt.plot(X, Y, 'kx')
    plt.plot(xx, samples[:, :, 0].numpy().T, 'C0', linewidth=.5)
    plt.ylim(-2., +2.)
    plt.xlim(0, 10)
    plt.close()
    return fig
