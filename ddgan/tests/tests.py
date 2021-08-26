"""
Copyright: Jón Atli Tómasson

Author: Jón Atli Tómasson, Zef Wolffs

Description: Pattern matching within the latent space
             Includes a DD and a non-DD version

Github Repository: https://github.com/acse-jat20/DD-GAN/
"""
from pytest import fixture
import numpy as np
import sklearn.preprocessing


@fixture(scope='module')
def ddgan():
    """
    Import ddgan module

    Returns:
        Module: The ddgan module
    """
    import ddgan
    return ddgan


@fixture(scope='module')
def gan(ddgan):
    """
    Create GAN

    Args:
        ddgan (Module): The ddgan module

    Returns:
        GAN: The GAN
    """

    kwargs = {
        "nsteps": 10,
        "ndims": 5,
        "batch_size": 40,
        "batches": 5,
        "seed": 143,
        "epochs": 2,
        "n_critic": 10,
        "gen_learning_rate": 5e-4,
        "disc_learning_rate": 5e-4,
    }

    gan = ddgan.GAN(**kwargs)
    gan.setup(find_old_model=False)
    ddgan.set_seed(gan.seed)

    return gan


@fixture(scope='module')
def optimize(gan, ddgan):
    """
    Create optimizer

    Args:
        gan (GAN): The GAN to optimize
        ddgan (Module): The ddgan module

    Returns:
        Optimizer: The optimizer
    """
    kwargs_opt = {
        "start_from": 100,
        "nPOD": 5,
        "nLatent": 10,
        "npredictions": 2,
        "optimizer_epochs": 11,
        "gan": gan
    }

    optimizer = ddgan.Optimize(**kwargs_opt)

    return optimizer


@fixture(scope='module')
def load_data(gan):
    """
    Load in data for running the GAN and Optimizer classes

    Args:
        gan (GAN): The GAN to load data for

    Returns:
        tuple: Variables related to input data
    """
    csv_data = np.load('./data/processed/Single/pod_coeffs_field_Velocity.npy')
    csv_data = csv_data[0, :, :]

    csv_data = np.float32(csv_data.T)
    csv_data = csv_data[300:600, :5]

    scaling = sklearn.preprocessing.MinMaxScaler(feature_range=[-1, 1])
    csv_data = scaling.fit_transform(csv_data)

    t_begin = 0
    t_end = 200

    training_data = np.zeros(
        (t_end, gan.ndims * gan.nsteps),
        dtype=np.float32)

    for step in range(gan.nsteps):
        training_data[:,
                      step*gan.ndims:(step+1)*gan.ndims
                      ] = csv_data[t_begin+step:t_end+step, :]

    return training_data, gan.ndims, scaling


def test_set_seed(ddgan):
    """
    Test that setting a seed works as expected

    Args:
        ddgan (module): ddgan module with all functions
    """
    ddgan.set_seed(42)

    # We can only retrieve the numpy random seed
    assert np.random.get_state()[1][0] == 42


def test_random_generator(ddgan):
    """
    Test that setting a seed works as expected

    Args:
        ddgan (module): ddgan module with all functions
    """
    normal_generator = ddgan.truncated_normal(low=-2., upp=2.)
    samples = normal_generator(1000)
    assert np.max(samples) <= 2.
    assert np.min(samples) >= -2.
    assert np.abs(np.mean(samples)) < 0.1


def test_gan_setup(ddgan):
    """
    Test the setup function of the GAN class

    Args:
        ddgan (module): ddgan module with all functions
    """
    kwargs = {
        "nsteps": 10,
        "ndims": 10,
        "nLatent": 10,
        "batch_size": 10,
        "batches": 10,
        "seed": 143,
        "epochs": 100
        }

    gan = ddgan.GAN(**kwargs)

    # Making sure all of the dictionary is correctly set on the class
    for att_name, att in kwargs.items():
        assert att == getattr(gan, att_name)


def test_train_gan(load_data, gan):
    """
    Little test to make sure the training procedure works. We can't really
    test for any exact values as changing one of any of the parameters of the
    network would probably significantly influence the results. Therefore we do
    multiple weaker tests.

    Args:
        load_data (tuple): Variables related to input data
        gan (GAN): The GAN itself
    """
    training_data, _, _ = load_data

    g_layers_pre = gan.generator.layers[0].get_weights()[0]
    d_layers_pre = gan.discriminator.layers[0].get_weights()[0]

    gan.learn_hypersurface_from_POD_coeffs(training_data)

    # Make sure output types are correct

    assert type(gan.d_loss.result().numpy()) == np.float32
    assert type(gan.d_loss.result().numpy()) == np.float32
    assert type(gan.d_loss.result().numpy()) == np.float32

    # We can also make sure the loss values are not bogus

    assert 1e10 > gan.d_loss.result().numpy() > -100
    assert 1e10 > gan.d_loss.result().numpy() > -100
    assert 1e10 > gan.d_loss.result().numpy() > -100

    # Let's also make sure the weights have actually changed since the start
    # and supposedly some training has found place

    assert not np.allclose(g_layers_pre,
                           gan.generator.layers[0].get_weights()[0])
    assert not np.allclose(d_layers_pre,
                           gan.discriminator.layers[0].get_weights()[0])


def test_optimize_gan(gan, optimize, load_data):
    """
    Test the optimization part of the GAN. Again, We can't really
    test for any exact values as changing one of any of the parameters of the
    network would probably significantly influence the results. Therefore we do
    multiple weaker tests.

    Args:
        gan (GAN): The GAN
        optimize (Optimize): The optimizer
        load_data (tupe): Variables related to input data
    """

    training_data, _, scaling = load_data

    flds = optimize.predict(training_data)

    assert flds.shape == (
        gan.nsteps + optimize.npredictions-1,
        gan.ndims
        )
