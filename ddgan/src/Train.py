"""
Jón Atli Tómasson and Zef Wolffs
"""

__all__ = ['GAN']

import glob
import tensorflow as tf
import numpy as np
import sklearn
import datetime
from dataclasses import dataclass
from ddgan.src.Utils import train_step


@dataclass(unsafe_hash=True)
class GAN:
    """
    Class for the predictive GAN
    """
    # Keyword argument definitions
    nsteps: int = 5  # Consecutive timesteps
    ndims: int = 5  # Reduced dimensions
    n_critic: int = 5
    lmbda: int = 10  # Gradient penalty multiplier
    batch_size: int = 20  # 32
    batches: int = 10  # 900
    seed: int = 143
    epochs: int = 500
    logs_location: str = './logs/gradient_tape/'

    # Objects
    generator = None
    discriminator = None
    generator_opt = None
    discriminator_opt = None

    # Losses
    g_loss = None
    d_loss = None
    w_loss = None

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
        self.generator_opt = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                                      beta_1=0,
                                                      beta_2=0.9)
        self.discriminator_opt = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                                          beta_1=0,
                                                          beta_2=0.9)
        self.make_logs()
        self.make_GAN()

    def make_logs(self) -> None:
        """
        Printing summaries for generator, discriminator and w in
        self.logs.location
        """
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
                                5, input_shape=(self.ndims, ),
                                activation='relu',
                                kernel_initializer=self.initializer))  # 5

        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.Dense(
                                10, activation='relu',
                                kernel_initializer=self.initializer))  # 10

        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.Dense(
                                (self.ndims*self.nsteps), activation='relu',
                                kernel_initializer=self.initializer))  # 25

        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.Dense(
                                (self.ndims*self.nsteps), activation='tanh',
                                kernel_initializer=self.initializer))  # 25

    def make_discriminator(self) -> None:
        """
        Create the discriminator network
        """
        self.discriminator = tf.keras.Sequential()
        self.discriminator.add(tf.keras.layers.Dense(
                                self.ndims*self.nsteps,
                                input_shape=(self.ndims*self.nsteps,),
                                kernel_initializer=self.initializer))

        self.discriminator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.discriminator.add(tf.keras.layers.Dropout(0.3))
        self.discriminator.add(tf.keras.layers.Dense(
                                10, kernel_initializer=self.initializer))

        self.discriminator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.discriminator.add(tf.keras.layers.Dense(
                                5, kernel_initializer=self.initializer))

        self.discriminator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.discriminator.add(tf.keras.layers.Dropout(0.3))
        self.discriminator.add(tf.keras.layers.Flatten())
        self.discriminator.add(tf.keras.layers.Dense(
                                1, kernel_initializer=self.initializer))

    def make_GAN(self, folder='models/') -> None:
        """
        Searching for an existing model, creating one from scratch
        if not found.
        """
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
        return tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)

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
        return -tf.reduce_mean(d_fake)

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
            tf.summary.scalar('loss', self.g_loss.result(), step=epoch)

        with self.d_summary_writer.as_default():
            tf.summary.scalar('loss', self.d_loss.result(), step=epoch)

        with self.w_summary_writer.as_default():
            tf.summary.scalar('loss', self.w_loss.result(), step=epoch)

    def resetting_states(self) -> None:
        """
        Resetting model loss states
        """
        self.g_loss.reset_states()
        self.d_loss.reset_states()
        self.w_loss.reset_states()

    def print_loss(self) -> None:
        """
        Printing the loss results
        """
        print('gen loss: ', self.g_loss.result().numpy(),
              'd loss: ', self.d_loss.result().numpy(),
              'w_loss: ', self.w_loss.result().numpy())

    def train(self,
              training_data: np.ndarray,
              gan_input: np.ndarray,
              ) -> None:
        """
        Training the GAN

        Args:
            training_data (np.ndarray): Actual values for comparison
            gan_input(np.ndarray): Random input values in the shape n x Dims
        """
        losses = np.zeros((self.epochs, 4))

        for epoch in range(self.epochs):
            print('epoch: \t', epoch)
            # real_data = training_data  # X1.astype('int')

            # uncommenting this line means that the noise is not paired with
            # the outputs (probably desirable)
            # noise = np.random.normal(
            # size=[gan_input.shape[0], gan_input.shape[1]]
            # )

            # shuffle each epoch
            real_data, noise = sklearn.utils.shuffle(training_data, gan_input)
            # shaped_real_data = real_data.reshape(
            #                         self.batches,
            #                         self.batch_size,
            #                         self.ndims*self.nsteps)

            shaped_real_data = real_data.reshape(
                                    10,
                                    self.batch_size,
                                    self.ndims*self.nsteps)

            shaped_noise = noise.reshape(
                                    10,
                                    self.batch_size,
                                    self.ndims)

            for i in range(shaped_real_data.shape[0]):
                train_step(self, shaped_noise[i], shaped_real_data[i])

            self.print_loss()

            losses[epoch, :] = [epoch+1,
                                self.g_loss.result().numpy(),
                                self.d_loss.result().numpy(),
                                self.w_loss.result().numpy()]

            self.write_summary(epoch)
            self.resetting_states()

            if (epoch + 1) % 1000 == 0:
                self.save_gan(epoch)

        # Saving the loss data in a csv file
        np.savetxt('losses.csv', losses, delimiter=',')

    def learn_hypersurface_from_POD_coeffs(self,
                                           nPOD,
                                           gan_input,
                                           training_data,
                                           ndims_latent_input):
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
        self.train(training_data, gan_input)
        print('ending training')

        # generate some random inputs and put through generator
        number_test_examples = 10
        test_input = tf.random.normal([number_test_examples,
                                       ndims_latent_input])
        predictions = self.generator(test_input, training=False)
        # predictions = generator.predict(test_input)
        # number_test_examples by ndims_latent_input

        predictions_np = predictions.numpy()  # nExamples by nPOD*nsteps
        # tf.compat.v1.InteractiveSession()
        # predictions_np = predictions.numpy().

        # Reshaping the GAN output (in order to apply inverse scaling)
        predictions_np = predictions_np.reshape(
            number_test_examples*self.nsteps,
            nPOD)

        return predictions_np
