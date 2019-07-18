import sys
import pytest

sys.path.append('./examples')


# from train_mnist import ex

@pytest.mark.examples
def test_train_mnist():
    """
    Just check whether the train_mist script still runs...

    NB: This script will fail if you test twice within the same minute, due to
    the name where output files are dumped being the same. Not a problem IRL.
    """
    # TODO: Run this test! Currently this doesn't run on CI, due to a bug in some metadata that TF should be giving
    #       as part of importing itself. This makes it not work with sacred. A PR has been merged into TF, but this
    #       hasn't made it into a release yet. https://github.com/tensorflow/tensorflow/issues/30028
    # ex.run(None, config_updates={"iterations": 2, "hz": {"short": 1}, "float_type": "float32", "jitter_level": 1e-4,
    #                              "num_inducing_points": 100, "dataset": "mnist-small-subset-for-test"})
