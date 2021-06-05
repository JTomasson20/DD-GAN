import numpy as np
import tensorflow as tf
import random

__all__ = ['set_seed']


def set_seed(seed):
    """
    Sets seed for numpy and tensorflow

    Args:
        seed (int): random number generator seed
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
