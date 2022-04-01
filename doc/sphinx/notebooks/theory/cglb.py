from pathlib import Path
import numpy as np


def load_snelson_data():
    curdir = Path(__file__).parent.resolve()
    train_inputs = "snelson_train_inputs.dat"
    train_outputs = "snelson_train_outputs.dat"
    datapath = lambda name: str(Path(curdir, "data", name).resolve())
    xfile = datapath(train_inputs)
    yfile = datapath(train_outputs)
    x = np.loadtxt(xfile).reshape(-1, 1)
    y = np.loadtxt(yfile).reshape(-1, 1)
    return (x, y)


def plot_prediction(ax, x, y, xnew, loc, scale, color, label):
    for k in (1, 2):
        lb = (loc - k * scale).squeeze()
        ub = (loc + k * scale).squeeze()
        alpha = 1 - 0.4 * k
        ax.fill_between(xnew, lb, ub, color=color, alpha=alpha)

    ax.plot(xnew, loc, color=color, label=label)
    ax.scatter(x, y, color="gray", s=8, alpha=0.7)
