

def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out
    any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return True