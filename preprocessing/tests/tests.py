import sys
import os
import numpy as np
import pytest
sys.path.insert(1, './../src/')
from get_pod_coeffs import get_pod_coeffs # noqa F401


@pytest.fixture(autouse=True)
def cleanup():
    # Quick check that the data is available. If not, test makes no sense.
    assert os.path.isfile(
        './../../data/FPC_Re3900_2D_CG_old/fpc_2D_Re3900_CG_0.vtu'), \
            "Data files not available! Make sure they are present"
    yield
    os.remove('./pod_coeffs_field_Velocity.npy')


def test_get_pod_coeffs():
    get_pod_coeffs(out_dir='.')
    coeffs = np.load('pod_coeffs_field_Velocity.npy')
    assert coeffs.shape == (4, 10, 200)

    # Regression test on data calculated earlier
    coeffs_correct = np.load('./test_data/pod_coeffs_field_Velocity.npy')

    assert (coeffs == coeffs_correct).all
