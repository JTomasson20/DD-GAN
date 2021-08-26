from pytest import fixture
import numpy as np
import tensorflow as tf
import sklearn.preprocessing

"""
The following tests test the DD version of the DD-GAN

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
    Create DD-GAN

    Args:
        ddgan (Module): The ddgan module

    Returns:
        GAN: The GAN
    """
    # Number of neighbour and self time values.
    # Corresponds to n = n_b, n_a, n_self
    added_dims = [1, 1, 3]

    kwargs = {
        "nsteps": np.sum(added_dims),
        "ndims": 10,
        "nLatent": 100,
        "batch_size": 20,
        "batches": 10,
        "seed": 143,
        "epochs": 2
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
        "start_from": 0,
        "nPOD": 10,
        "nLatent": 100,
        "npredictions": 2,
        "optimizer_epochs": 10,
        "dt": 1,
        "gan": gan,
        "cycles": 2,
        "disturb": True,
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

    # Number of neighbour and self time values.
    # Corresponds to n = n_b, n_a, n_self
    added_dims = [1, 1, 3]
    # Cumulative sum
    cumulative_dims = [0, 1, 2]

    csv_data = np.load('./data/processed/DD/pod_coeffs_field_Velocity.npy')

    subdomains = csv_data.shape[0]
    nPOD = csv_data.shape[1]
    datapoints = csv_data.shape[2]

    assert 10 <= nPOD, "Make sure the data includes enough POD coeffs"
    assert 200 <= datapoints, "Not enough data"
    assert 4 <= subdomains, "Not enough domains"

    csv_data = csv_data[:, :10, :300]
    nPOD = 10
    tmp_data = np.ones([subdomains, 300, nPOD])

    for k in range(subdomains):
        tmp_data[k] = csv_data[k].T

    csv_data = tmp_data

    scales = []
    scaled_training = np.zeros_like(csv_data)
    for i in range(subdomains):
        scales.append(
            sklearn.preprocessing.MinMaxScaler(feature_range=[-1, 1])
            )
        scaled_training[i] = scales[i].fit_transform(csv_data[i])

    assert sum(added_dims[:-1]) == cumulative_dims[-1], 'Check added_dims'

    training_data = np.zeros(
            (2, 100, np.sum(added_dims * nPOD)
             ), dtype=np.float32
        )
    # See image above on how the data is structured
    for domain in range(2):
        for i, dim in enumerate([0, 2, 1]):
            for j, step in enumerate(range(0, 3)):
                training_data[
                    domain, :,
                    cumulative_dims[i]*nPOD + j*nPOD:
                    cumulative_dims[i]*nPOD + (j+1)*nPOD
                    ] = scaled_training[dim + domain][step:100 + step, :]

    # Adding data for leftmost and rightmost domain
    boundrary_conditions = []
    boundrary_conditions.append(scaled_training[0, 2:])
    boundrary_conditions.append(scaled_training[-1, 2:])

    joined_train_data = training_data.reshape(
        (training_data.shape[1]*training_data.shape[0], training_data.shape[2])
        )
    return np.float32(joined_train_data), boundrary_conditions


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
        "batch_size": 20,
        "batches": 10,
        "seed": 143,
        "epochs": 100
        }

    gan = ddgan.GAN(**kwargs)

    # Making sure all of the dictionary is correctly set on the class
    for att_name, att in kwargs.items():
        assert att == getattr(gan, att_name)


def test_optimize_DD_gan(gan, optimize, load_data):
    """
    Test the optimization part of the GAN. Again, We can't really
    test for any exact values as changing one of any of the parameters of the
    network would probably significantly influence the results.
    Therefore we do
    multiple weaker tests.

    Args:
        gan (GAN): The GAN
        optimize (Optimize): The optimizer
        load_data (tupe): Variables related to input data
    """

    training_data, boundrary = load_data

    flds = optimize.predict(
        training_data,
        tf.convert_to_tensor(np.array(boundrary, dtype=np.float32))
        )

    assert flds.shape == (
        gan.nsteps + optimize.npredictions-1,
        gan.ndims
        )
