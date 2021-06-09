import numpy as np
import tensorflow as tf
import random

__all__ = ['set_seed', 'train_step']


def set_seed(seed):
    """
    Sets seed for numpy and tensorflow

    Args:
        seed (int): random number generator seed
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


@tf.function
def train_step(gan, noise: np.ndarray, real: np.ndarray) -> None:
    """
    Training the gan for a single step

    Args:
        noise (np.ndarray): gaussian noise input
        real (np.ndarray): actual values
    """
    for i in range(gan.n_critic):
        with tf.GradientTape() as t:
            with tf.GradientTape() as t1:
                fake = gan.generator(noise, training=True)
                epsilon = tf.random.uniform(shape=[gan.batch_size, 1], minval=0.,
                                            maxval=1.)

                interpolated = real + epsilon * (fake - real)
                t1.watch(interpolated)
                c_inter = gan.discriminator(interpolated, training=True)                    
                d_real = gan.discriminator(real, training=True)
                d_fake = gan.discriminator(fake, training=True)
                d_loss = gan.discriminator_loss(d_real, d_fake)

            grad_interpolated = t1.gradient(c_inter, interpolated)
            slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_interpolated) + 1e-12,
                                           axis=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            new_d_loss = d_loss + (gan.lmbda*gradient_penalty)

        c_grad = t.gradient(new_d_loss,
                            gan.discriminator.trainable_variables)
        gan.discriminator_opt.apply_gradients(zip(c_grad,
                                                  gan.discriminator.
                                                  trainable_variables))

    # train generator
    with tf.GradientTape() as gen_tape:
        fake_images = gan.generator(noise, training=True)
        d_fake = gan.discriminator(fake_images, training=True)
        g_loss = gan.generator_loss(d_fake)

    gen_grads = gen_tape.gradient(g_loss,
                                  gan.generator.trainable_variables)

    gan.generator_opt.apply_gradients(zip(gen_grads,
                                          gan.generator.
                                          trainable_variables))

    # for tensorboard
    gan.g_loss(g_loss)
    gan.d_loss(new_d_loss)
    gan.w_loss((-1)*(d_loss))  # wasserstein distance


    
def mse_loss(inp, outp):
    """
    Wrapper for mean square error loss of inp v outp
    """
    return tf.keras.losses.MeanSquaredError(inp, outp)
