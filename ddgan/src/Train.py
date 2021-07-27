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
from ddgan.src.Utils import train_step, truncated_normal


@dataclass(unsafe_hash=True)
class GAN:
    """
    Class for the predictive GAN
    """
    # Keyword argument definitions
    # Input data parameters
    nsteps: int = 5  # Consecutive timesteps
    ndims: int = 5  # Reduced dimensions
    batch_size: int = 20  # 32
    batches: int = 10  # 900
    seed: int = 143  # Random seed for reproducability
    epochs: int = 500  # Number of training epochs
    nLatent: int = 10  # Dimensionality of the latent space

    # WGAN training parameters
    n_critic: int = 5  # Number of gradient penalty computations per epoch
    lmbda: int = 10  # Gradient penalty multiplier

    gen_learning_rate: float = 0.0001  # Generator optimization learning rate
    disc_learning_rate: float = 0.0001  # Discriminator optimization learning
    noise: bool = False  # If input generator_input is added during training

    # Objects - Can be filled in at bootup and skip calling setup
    # Remember to make logs if doing so
    generator = None
    discriminator = None
    generator_opt = None
    discriminator_opt = None

    # Losses
    g_loss = None
    d_loss = None
    w_loss = None

    # Logs
    logs_location: str = './logs/gradient_tape/'
    model_location: str = 'models/'

    # Other definitions
    g_summary_writer = None
    d_summary_writer = None
    w_summary_writer = None

    initializer = tf.keras.initializers.RandomNormal(mean=0.0,
                                                     stddev=0.05,
                                                     seed=seed)

    random_generator = truncated_normal(mean=0, sd=1, low=-6, upp=6)

    # Only in use if noise = True
    noise_generator = None
    # Standard deviation of a normal distribution with mu = 0
    noise_level: float = 0.02

    def setup(self, find_old_model=False) -> None:
        """
        Setting up the neccecary values for the GAN class

        Args:
            find_old_model (bool): whether to look for a ready model
                in stead of building one from scratch`
        """
        self.generator_opt = tf.keras.optimizers.Nadam(
            learning_rate=self.gen_learning_rate,
            beta_1=0,
            beta_2=0.9
            )

        self.discriminator_opt = tf.keras.optimizers.Nadam(
            learning_rate=self.disc_learning_rate,
            beta_1=0,
            beta_2=0.9
            )

        # Data augmentation noise
        if self.noise:
            self.noise_generator = truncated_normal(mean=0,
                                                    sd=self.noise_level)
        else:
            self.noise_generator = np.zeros

        self.make_GAN(find_old_model=find_old_model)

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
                                10, input_shape=(self.nLatent, ),
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

    def make_GAN(self, find_old_model=False) -> None:
        """Searching for an existing model, creating one from scratch
        if not found.

        Args:
            find_old_model (bool, optional): To search for
                an existing model at self.model_location Defaults to False.
        """
        if find_old_model:
            try:
                print('looking for previous saved models')
                g_dir = glob.glob('./' + self.model_location + 'saved_g_*')
                d_dir = glob.glob('./' + self.model_location + 'saved_c_*')

                if g_dir and d_dir:
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

        else:
            print('making new generator and critic')
            self.make_generator()
            self.make_discriminator()

    def discriminator_loss(self,
                           d_real: np.ndarray,
                           d_fake: np.ndarray
                           ) -> float:
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

    def generator_loss(self, d_fake: np.ndarray) -> float:
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
            folder (string): model location. Defaults to
                ´models´
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
              ) -> None:
        """
        Training the GAN

        Args:
            training_data (np.ndarray): Actual values for comparison
        """
        losses = np.zeros((self.epochs, 4))

        for epoch in range(self.epochs):
            print('epoch: \t', epoch)

            # Drawing samples from normal distribution for
            # generator
            generator_input = self.random_generator(
                [training_data.shape[0], self.nLatent]
                )

            # Shuffle each epoch. Command does so individually
            real_data, generator_input = sklearn.utils.shuffle(
                training_data, generator_input
                )

            # Reshaping GAN inputs
            shaped_real_data = real_data.reshape(
                                    self.batches,
                                    self.batch_size,
                                    self.ndims*self.nsteps)

            shaped_generator_input = generator_input.reshape(
                                    self.batches,
                                    self.batch_size,
                                    self.nLatent)
            # Single epoch
            for i in range(shaped_real_data.shape[0]):
                train_step(self,
                           shaped_generator_input[i],
                           shaped_real_data[i] +
                           self.noise_generator(
                               shaped_real_data[i].shape
                               ).astype(np.float32))

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

    def learn_hypersurface_from_POD_coeffs(
            self,
            training_data: np.ndarray):
        """
        Umbrella function that makes logs and begins
            training

        Args:
            training_data (np.ndarray): [description]

        Returns:
            (tensor): Fitted data and future predictions
        """
        # logs to follow losses on tensorboard
        self.make_logs()

        print('beginning training')
        self.train(training_data)
        print('ending training')

        return None

    def resume_training(self, training_data, start_epoch):
        """Umbrella function for resuming training from
                starting epoch.

        Args:
            training_data (np.ndarray): Structured training
                data in 1d array
            start_epoch (int): epoch to resume from
        """
        for epoch in range(start_epoch, self.epochs):
            print('epoch: \t', epoch)

            generator_input = self.random_generator(
                [training_data.shape[0], self.nLatent]
                )

            # shuffle each epoch
            real_data, generator_input = \
                sklearn.utils.shuffle(training_data, generator_input)

            shaped_real_data = real_data.reshape(
                                    self.batches,
                                    self.batch_size,
                                    self.ndims*self.nsteps)

            shaped_generator_input = generator_input.reshape(
                                    self.batches,
                                    self.batch_size,
                                    self.nLatent)

            for i in range(shaped_real_data.shape[0]):
                train_step(self,
                           shaped_generator_input[i],
                           shaped_real_data[i] +
                           self.noise_generator(shaped_real_data.shape))

            self.print_loss()
            self.write_summary(epoch)
            self.resetting_states()

            if (epoch + 1) % 1000 == 0:
                self.save_gan(epoch)
