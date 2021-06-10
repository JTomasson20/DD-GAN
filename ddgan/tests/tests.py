from pytest import fixture
import numpy as np
import tensorflow as tf
import sklearn.preprocessing

"""
Please execute module from root of repository
"""


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
        "nsteps": 5,
        "ndims": 5,
        "lmbda": 10,
        "n_critic": 5,
        "batches": 10,
        "batch_size": 20,
        "seed": 143
    }

    gan = ddgan.GAN(**kwargs)
    gan.setup()
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
        "initial": 5,
        "inn": 10,
        "iterations": 20,
        "optimizer_epochs": 5000,
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
    csv_data = np.loadtxt('./data/processed/POD_coeffs_1_204_orig.csv',
                          delimiter=',')
    csv_data = np.float32(csv_data)

    nTrain = csv_data.shape[1]
    nPOD = csv_data.shape[0]

    csv_data = csv_data.T  # nTrain by nPOD

    scaling = sklearn.preprocessing.MinMaxScaler(feature_range=[-1, 1])
    csv_data = scaling.fit_transform(csv_data)

    t_begin = 0
    t_end = nTrain - gan.nsteps + 1

    training_data = np.zeros((t_end, nPOD * gan.nsteps), dtype=np.float32)

    input_to_GAN = tf.random.normal([training_data.shape[0],
                                     gan.ndims])
    input_to_GAN = input_to_GAN.numpy()

    for step in range(gan.nsteps):
        training_data[:, step*nPOD:(step+1)*nPOD] = csv_data[t_begin+step:
                                                             t_end+step, :]

    return training_data, input_to_GAN, nPOD, scaling


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
    training_data, input_to_GAN, nPOD, _ = load_data

    g_layers_pre = gan.generator.layers[0].get_weights()[0]
    d_layers_pre = gan.discriminator.layers[0].get_weights()[0]

    gan.learn_hypersurface_from_POD_coeffs(nPOD, input_to_GAN, training_data,
                                           5, epochs=2)

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

    training_data, _, nPOD, scaling = load_data
    nLatent = 5

    start_from = 100
    inn = training_data[start_from,
                        :(gan.nsteps-1)*nPOD].reshape(1, (gan.nsteps - 1) *
                                                      nLatent)

    npredictions = 2

    initial_comp = training_data[start_from,
                                 :(gan.nsteps - 1)*nPOD].reshape((gan.nsteps -
                                                                  1), nLatent)
    flds = optimize.timesteps(initial_comp, inn, npredictions)

    flds = scaling.inverse_transform(flds).T

    assert type(flds) == np.ndarray
