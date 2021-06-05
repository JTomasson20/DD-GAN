from pytest import fixture
import numpy as np


@fixture(scope='module')
def ddgan():
    import ddgan 
    return ddgan


def test_set_seed(ddgan):
    ddgan.set_seed(42)

    # We can only retrieve the numpy random seed
    assert np.random.get_state()[1][0] == 42
