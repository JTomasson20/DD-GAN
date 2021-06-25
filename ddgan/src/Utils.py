import numpy as np
import tensorflow as tf
import random

__all__ = ['set_seed', 'train_step']


def set_seed(seed):
    """
    Sets seed for random, numpy and tensorflow

    Args:
        seed (int): Random number generator seed
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# Clashes when trying to include this within the GAN class
@tf.function
def train_step(gan, batch: np.ndarray) -> None:
    """
    Training the gan for a single step

    Args:
        gan (GAN): Model object
        noise (np.ndarray): Gaussian noise input
    """
    noise = tf.random.normal([gan.batch_size, gan.latent_space])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gan.generator(noise, training=True)

        real_output = gan.discriminator(batch, training=True)
        fake_output = gan.discriminator(generated_images, training=True)

        gen_loss = gan.generator_loss(fake_output)
        disc_loss = gan.discriminator_loss(real_output, fake_output)

    gen_grads = gen_tape.gradient(
        gen_loss,
        gan.generator.trainable_variables
        )
    disc_grads = disc_tape.gradient(
        disc_loss,
        gan.discriminator.trainable_variables
        )

    gan.generator_opt.apply_gradients(
        zip(gen_grads, gan.generator.trainable_variables)
        )
    gan.discriminator_opt.apply_gradients(
        zip(disc_grads, gan.discriminator.trainable_variables)
        )

    gan.generator_mean_loss(gen_loss)
    gan.discriminator_mean_loss(disc_loss)

