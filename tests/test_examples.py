import sys

sys.path.append('./examples')

from train_mnist import ex


def test_train_mnist():
    """
    Just check whether the train_mist script still runs...

    NB: This script will fail if you test twice within the same minute, due to
    the name where output files are dumped being the same. Not a problem IRL.
    """
    # ex.run("print_config", config_updates={"iterations": 3, "hz": {"short": 1}})
    ex.run(None, config_updates={"iterations": 2, "hz": {"short": 1}, "float_type": "float32", "jitter_level": 1e-4,
                                 "num_inducing_points": 100, "dataset": "mnist-small-subset-for-test"})
