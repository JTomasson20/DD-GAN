import time
import tensorflow as tf
import sklearn.utils
import sklearn.preprocessing
import datetime
import numpy as np

if __name__ == "__main__":
    from ddgan.src.Train import GAN
    from ddgan.src.Utils import *

    kwargs = {
        "nsteps" : 5,
        "ndims" : 5,
        "lambda" : 10,
        "n_critic" : 5,
        "batches" : 10,
        "batch_size" : 20
    }
    gan = GAN()
    gan.setup(kwargs)

    nTrain = 100
    nPOD = 10

    t_begin = 0
    t_end = nTrain - gan.nsteps + 1
    training_data = np.zeros((t_end, nPOD * gan.nsteps), dtype=np.float32) # nTrain by nsteps*nPOD # 'float32' or np.float32

    input_to_GAN = tf.random.normal([training_data.shape[0], ndims_latent_input])
    input_to_GAN = input_to_GAN.numpy()

    


