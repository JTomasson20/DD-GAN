"""
Copyright: Jón Atli Tómasson

Author: Jón Atli Tómasson

Description: Utility functions for the DD-GAN

Github Repository: https://github.com/acse-jat20/DD-GAN/
"""
import numpy as np
import tensorflow as tf
import random
from scipy.stats import truncnorm

__all__ = ['set_seed', 'train_step', 'truncated_normal']


def set_seed(seed):
    """
    Sets seed for random, numpy and tensorflow

    Args:
        seed (int): Random number generator seed
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def truncated_normal(mean=0., sd=1., low=-4., upp=4.):
    """
    Generating a trunicated scipy random number generator

    Args:
        mean (float, optional): mean of the distribution. Defaults to 0.
        sd (float, optional): standard deviation. Defaults to 1.
        low (float, optional): lower bound. Defaults to -5.
        upp (float, optional): upper bound. Defaults to 5.

    Returns:
        scipy.stats obj: trunicated normal distribution rng
    """
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd
        ).rvs


# Clashes when trying to include this within the GAN class
@tf.function
def train_step(gan,
               noise: np.ndarray,
               real: np.ndarray,
               reverse_step: bool = False
               ) -> None:
    """
    Training the gan for a single step

    Args:
        gan (GAN): Model object
        noise (np.ndarray): Gaussian noise input
        real (np.ndarray): Actual values
        reverse_step(bool): Whether to make the discriminator take a step back
    """
    for i in range(gan.n_critic):
        with tf.GradientTape() as t:
            with tf.GradientTape() as t1:
                fake = gan.generator(
                    tf.random.normal(
                        shape=noise.shape),
                    training=True)

                epsilon = tf.random.uniform(shape=[gan.batch_size, 1],
                                            minval=0., maxval=1.)

                interpolated = fake + epsilon * (real - fake)
                t1.watch(interpolated)
                c_inter = gan.discriminator(interpolated, training=True)
                d_real = gan.discriminator(real, training=True)
                d_fake = gan.discriminator(fake, training=True)
                if reverse_step:
                    d_loss = gan.discriminator_loss(d_fake, d_real)
                else:
                    d_loss = gan.discriminator_loss(d_real, d_fake)

            grad_interpolated = t1.gradient(c_inter, interpolated)
            slopes = tf.sqrt(tf.reduce_sum(
                            tf.square(grad_interpolated) + 1e-12, axis=[1])
                            )

            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            new_d_loss = d_loss + (gan.lmbda*gradient_penalty)

        c_grad = t.gradient(new_d_loss,
                            gan.discriminator.trainable_variables
                            )

        gan.discriminator_opt.apply_gradients(
                            zip(c_grad, gan.discriminator.trainable_variables)
                            )

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
