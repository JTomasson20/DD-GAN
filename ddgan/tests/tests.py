from pytest import fixture
import numpy as np


@fixture(scope='module')
def ddgan():
    import ddgan
    return ddgan


def test_set_seed(ddgan):
    """
    Test that setting a seed works as expected

    Args:
        ddgan (module): ddgan module with all functions
    """
    ddgan.set_seed(42)

    # We can only retrieve the numpy random seed
    assert np.random.get_state()[1][0] == 42


def test_gan_setup(ddgan):
    """
    Test the setup function of the GAN class

    Args:
        ddgan (module): ddgan module with all functions
    """
    kwargs = {
        "nsteps": 5,
        "ndims": 5,
        "lmbda": 10,
        "n_critic": 5,
        "batches": 10,
        "batch_size": 20
        }

    gan = ddgan.GAN()
    gan.setup(**kwargs)

    # Making sure all of the dictionary is correctly set on the class
    for att_name, att in kwargs.items():
        assert att == getattr(gan, att_name)
