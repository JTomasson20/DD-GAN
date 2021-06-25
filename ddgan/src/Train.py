"""
Jón Atli Tómasson and Zef Wolffs
"""

__all__ = ['GAN']

import glob
import tensorflow as tf
import numpy as np
import datetime
from dataclasses import dataclass
from ddgan.src.Utils import train_step


@dataclass(unsafe_hash=True)
class GAN:
    """
    Class for the predictive GAN
    """
    # Keyword argument definitions
    nsteps: int = 20  # Consecutive timesteps
    ndims: int = 15  # Reduced dimensions
    n_critic: int = 5
    lmbda: int = 10  # Gradient penalty multiplier
    batch_size: int = 256  # 32
    batches: int = 10  # 900
    seed: int = 143
    epochs: int = 500
    logs_location: str = './logs/gradient_tape/'

    # Added definitions
    latent_space: int = 100

    # Objects
    generator = None
    discriminator = None
    generator_opt = None
    discriminator_opt = None

    # Losses
    g_loss = None
    d_loss = None
    w_loss = None
    cross_entropy = None
    generator_mean_loss = None
    discriminator_mean_loss = None

    # Other definitions
    g_summary_writer = None
    d_summary_writer = None
    w_summary_writer = None

    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05,
                                                     seed=seed)

    def setup(self) -> None:
        """
        Setting up the neccecary values for the GAN class

        Args:
            kwargs (dict): key-value pairs of input variables
        """
        self.generator_opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.discriminator_opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(
            from_logits=True
            )

        self.make_logs()
        self.make_GAN()

    def make_logs(self) -> None:
        """
        Printing summaries for generator, discriminator and w in
        self.logs.location
        """
        self.generator_mean_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.discriminator_mean_loss = tf.keras.metrics.Mean(dtype=tf.float32)

        self.g_loss = tf.keras.metrics.Mean('g_loss', dtype=tf.float32)
        self.d_loss = tf.keras.metrics.Mean('d_loss', dtype=tf.float32)
        self.w_loss = tf.keras.metrics.Mean('w_loss', dtype=tf.float32)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        g_log_dir = self.logs_location + current_time + '/g'
        d_log_dir = self.logs_location + current_time + '/d'
        w_log_dir = self.logs_location + current_time + '/w'

        self.g_summary_writer = tf.summary.create_file_writer(g_log_dir)
        self.d_summary_writer = tf.summary.create_file_writer(d_log_dir)
        self.w_summary_writer = tf.summary.create_file_writer(w_log_dir)

    def make_generator(self) -> None:
        """
        Create the generator network
        """
        self.generator = tf.keras.Sequential()
        self.generator.add(tf.keras.layers.Dense(
                                5*4*256, use_bias=False,
                                input_shape=(self.latent_space, )))  # ,
        # kernel_initializer=self.initializer))  # 5

        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU(0.2))

        self.generator.add(tf.keras.layers.Reshape((5, 4, 256)))
        self.generator.add(tf.keras.layers.Conv2DTranspose(
                                128, (3, 3),
                                strides=(1, 1),
                                padding='same',
                                use_bias=False))
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU(0.2))

        self.generator.add(tf.keras.layers.Conv2DTranspose(
                                64, (3, 3),
                                strides=(2, 2),
                                padding='same',
                                use_bias=False))
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU(0.2))

        self.generator.add(tf.keras.layers.Conv2DTranspose(
                                1, (3, 3),
                                strides=(2, 2),
                                padding='same',
                                output_padding=[1, 0],
                                use_bias=False,
                                activation='tanh'))
        self.generator.summary()

    def make_discriminator(self) -> None:
        """
        Create the discriminator network
        """
        self.discriminator = tf.keras.Sequential()
        self.discriminator.add(tf.keras.layers.Conv2D(
                                    64, (3, 3),
                                    strides=(2, 2),
                                    padding='same',
                                    input_shape=[20, 15, 1]))
        self.discriminator.add(tf.keras.layers.LeakyReLU(0.2))

        self.discriminator.add(tf.keras.layers.Conv2D(
                                    128, (5, 5),
                                    strides=(1, 2),
                                    padding='same'))
        self.discriminator.add(tf.keras.layers.LeakyReLU(0.2))

        self.discriminator.add(tf.keras.layers.Flatten())
        self.discriminator.add(tf.keras.layers.Dense(1))

    def make_GAN(self, folder='models/') -> None:
        """
        Searching for an existing model, creating one from scratch
        if not found.
        """
        try:
            print('looking for previous saved models')
            g_dir = glob.glob('./' + folder + 'saved_g_*')
            d_dir = glob.glob('./' + folder + 'saved_d_*')

            if g_dir and g_dir:
                self.generator = tf.keras.models.load_model(g_dir[-1])
                self.discriminator = tf.keras.models.load_model(d_dir[-1])
            else:
                print('making new generator and critic')
                self.make_generator()
                self.make_discriminator()

        except OSError:
            print('making new generator and critic')
            self.make_generator()
            self.make_discriminator()

    def discriminator_loss(self, d_real: float, d_fake: float) -> float:
        """
        Calculate the loss for the discriminator as the sum of the reduced
        real and fake discriminator losses

        Args:
            d_real (float): Discriminator loss form classifying real data
            d_fake (float): Discriminator loss form classifying fake data

        Returns:
            float: Discriminator loss
        """
        real_loss = self.cross_entropy(tf.ones_like(d_real), d_real)
        fake_loss = self.cross_entropy(tf.zeros_like(d_fake), d_fake)
        return real_loss + fake_loss

    def generator_loss(self, d_fake: float) -> float:
        """
        Calculate the loss of the generator as the negative reduced fake
        discriminator loss. The generator has the task of fooling the
        discriminator.

        Args:
            d_fake (float): Discriminator loss from classifying fake data

        Returns:
            float: Generator loss
        """

        return self.cross_entropy(tf.ones_like(d_fake), d_fake)

    def save_gan(self, epoch: int, folder='models/') -> None:
        """
        Saving a trained model

        Args:
            epoch (int): Epoch number
        """
        saved_g_dir = './' + folder + 'saved_g_' + str(epoch + 1)
        saved_d_dir = './' + folder + 'saved_c_' + str(epoch + 1)
        tf.keras.models.save_model(self.generator, saved_g_dir)
        tf.keras.models.save_model(self.discriminator, saved_d_dir)

    def write_summary(self, epoch: int) -> None:
        """
        Writing a summary from the current model state

        Args:
            epoch (int): Current epoch
        """
        with self.g_summary_writer.as_default():
            tf.summary.scalar(
                'loss',
                self.generator_mean_loss.result(),
                step=epoch)

        with self.d_summary_writer.as_default():
            tf.summary.scalar(
                'loss',
                self.discriminator_mean_loss.result(),
                step=epoch)

    def resetting_states(self) -> None:
        """
        Resetting model loss states
        """
        self.generator_mean_loss.reset_states()
        self.discriminator_mean_loss.reset_states()

    def print_loss(self) -> None:
        """
        Printing the loss results
        """
        print('generator loss: ',
              self.generator_mean_loss.result().numpy(),
              'discriminator loss: ',
              self.discriminator_mean_loss.result().numpy()
              )

    def train(self, dataset: np.ndarray) -> None:
        """
        Training the GAN

        Args:
            training_data (np.ndarray): Actual values for comparison
            gan_input(np.ndarray): Random input values in the shape n x Dims
        """
        losses = np.zeros((self.epochs, 3))

        for epoch in range(self.epochs):
            print('epoch: \t', epoch)

            for batch in dataset:
                train_step(self, batch)

            self.print_loss()

            losses[epoch, :] = [epoch+1,
                                self.generator_mean_loss.result().numpy(),
                                self.discriminator_mean_loss.result().numpy()]

            self.write_summary(epoch)
            self.resetting_states()

            if (epoch + 1) % 1000 == 0:
                self.save_gan(epoch)

        # Saving the loss data in a csv file
        np.savetxt('losses.csv', losses, delimiter=',')

    def learn_hypersurface_from_POD_coeffs(self,
                                           training_data):
        """
        Make and train a model

        Args:
            nPOD ([type]): [description]
            gan_input ([type]): [description]
            training_data ([type]): [description]
            ndims_latent_input ([type]): [description]

        Returns:
            [type]: [description]
        """
        # logs to follow losses on tensorboard
        self.make_logs()

        print('beginning training')
        self.train(training_data)
        print('ending training')
