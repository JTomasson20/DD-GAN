"""
Jón Atli Tómasson and Zef Wolffs
"""

__all__ = ['GAN']

import tensorflow as tf
import numpy as np
import sklearn
import datetime


class GAN:
    """
    Class for the predictive GAN
    """

    def __init__(self) -> None:
        """
        Predictive GAN class constructor
        """

        # Keyword argument definitions
        self.kwargs = None
        self.nsteps = None
        self.ndims = None
        self.lmbda = None
        self.n_critic = None
        self.batch_size = None  # 32
        self.batches = None  # 900

        # Method definitions
        self.generator = tf.keras.Sequential()
        self.discriminator = tf.keras.Sequential()
        self.generator_opt = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                                      beta_1=0, beta_2=0.9)
        self.discriminator_opt = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                                          beta_1=0, beta_2=0.9)
        self.g_loss = tf.keras.metrics.Mean('g_loss', dtype=tf.float32)
        self.d_loss = tf.keras.metrics.Mean('d_loss', dtype=tf.float32)
        self.w_loss = tf.keras.metrics.Mean('w_loss', dtype=tf.float32)

        # Other definitions
        self.g_summary_writer = None
        self.d_summary_writer = None
        self.w_summary_writer = None

    def setup(self, kwargs) -> None:
        """
        Setting up the neccecary values for the GAN class

        Args:
            kwargs (dict): key-value pairs of input variables
        """
        self.kwargs = kwargs
        # Number of consecutive timesteps
        self.nsteps = kwargs.pop("nsteps", 5)
        # Number of reduced variables
        self.ndims = kwargs.pop("ndims", 5)

        self.lmbda = kwargs.pop("lambda", 10)
        self.n_critic = kwargs.pop("n_critic", 5)
        self.batch_size = kwargs.pop("batch_size", 20)  # 32
        self.batches = kwargs.pop("batches", 10)  # 900
        self.make_logs()
        self.make_GAN()

    def make_logs(self) -> None:
        """
        Logging utility
        """
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        g_log_dir = './logs/gradient_tape/' + current_time + '/g'
        d_log_dir = './logs/gradient_tape/' + current_time + '/d'
        w_log_dir = './logs/gradient_tape/' + current_time + '/w'

        self.g_summary_writer = tf.summary.create_file_writer(g_log_dir)
        self.d_summary_writer = tf.summary.create_file_writer(d_log_dir)
        self.w_summary_writer = tf.summary.create_file_writer(w_log_dir)

    def make_generator(self) -> None:  # nsteps):
        """
        Create the generator network
        """
        self.generator.add(tf.keras.layers.Dense(5, input_shape=(5,),
                                                 activation='relu'))  # 5
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.Dense(10, activation='relu'))  # 10
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.Dense((5*self.nsteps),
                           activation='relu'))  # 25
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.Dense((5*self.nsteps),
                           activation='tanh'))  # 25

    def make_discriminator(self) -> None:  # nsteps):
        """
        Create the discriminator network
        """
        self.discriminator.add(
            tf.keras.layers.Dense(5*self.nsteps,
                                  input_shape=(5*self.nsteps,)))
        self.discriminator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.discriminator.add(tf.keras.layers.Dropout(0.3))
        self.discriminator.add(tf.keras.layers.Dense(10))
        self.discriminator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.discriminator.add(tf.keras.layers.Dense(5))
        self.discriminator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.discriminator.add(tf.keras.layers.Dropout(0.3))
        self.discriminator.add(tf.keras.layers.Flatten())
        self.discriminator.add(tf.keras.layers.Dense(1))

    def make_GAN(self, model_number=1) -> None:
        """
        Searching for an existing model, creating one from scratch
        if not found.
        """
        try:
            print('looking for previous saved models')
            saved_g1_dir = './saved_g_' + str(model_number)
            self.generator = tf.keras.models.load_model(saved_g1_dir)

            saved_d1_dir = './saved_c_' + str(model_number)
            self.discriminator = tf.keras.models.load_model(saved_d1_dir)

        except OSError:  # Add error type
            print('making new generator and critic')
            self.make_generator()
            self.make_discriminator()

    def discriminator_loss(self, d_real, d_fake) -> float:
        """
        Calculate the loss for the discriminator as the sum of the reduced
        real and fake discriminator losses

        Args:
            d_real (float): Discriminator loss form classifying real data
            d_fake (float): Discriminator loss form classifying fake data

        Returns:
            float: Discriminator loss
        """
        d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
        return d_loss

    def generator_loss(self, d_fake) -> float:
        """
        Calculate the loss of the generator as the negative reduced fake
        discriminator loss. The generator has the task of fooling the
        discriminator.

        Args:
            d_fake (float): Discriminator loss form classifying fake data

        Returns:
            float: Generator loss
        """
        g_loss = -tf.reduce_mean(d_fake)
        return g_loss

    def update_discriminator_loss(self, d_loss, fake, real) -> float:
        """
        Update the discriminator loss

        Args:
            d_loss (float): Discriminator loss
            batch_size (int): Batch size
            fake (np.array): Fake (generated) data
            real (np.array): Real data sampled from input

        Returns:
            Scalar: Loss with gradient penalty applied
        """
        with tf.GradientTape() as t:
            epsilon = tf.random.uniform(shape=[self.batch_size, 1], minval=0.,
                                        maxval=1.)
            interpolated = real + epsilon * (fake - real)
            t.watch(interpolated)
            c_inter = self.discriminator(interpolated, training=True)

        grad_interpolated = t.gradient(c_inter, interpolated)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_interpolated) + 1e-12,
                                       axis=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        return d_loss + (self.lmbda*gradient_penalty)

    def save_gan(self, epoch) -> None:
        """
        Saving a trained model

        Args:
            epoch (int): Epoch number
        """
        saved_g1_dir = './saved_g_' + str(epoch + 1)
        saved_d1_dir = './saved_c_' + str(epoch + 1)
        tf.keras.models.save_model(self.generator, saved_g1_dir)
        tf.keras.models.save_model(self.discriminator, saved_d1_dir)

    def write_summary(self, epoch) -> None:
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

    @tf.function
    def train_step(self, noise, real) -> None:
        """
        Training the gan for a single step

        Args:
            noise (np.array): gaussian noise input
            real (np.array): actual values
        """
        for i in range(self.n_critic):
            with tf.GradientTape() as t:
                fake = self.generator(noise, training=True)
                d_real = self.discriminator(real, training=True)
                d_fake = self.discriminator(fake, training=True)
                d_loss = self.discriminator_loss(d_real, d_fake)

            new_d_loss = self.update_discriminator_loss(d_loss,
                                                        self.batch_size,
                                                        fake,
                                                        real)
            c_grad = t.gradient(new_d_loss,
                                self.discriminator.trainable_variables)
            self.discriminator_opt.apply_gradients(zip(c_grad,
                                                       self.discriminator.
                                                       trainable_variables))

        # train generator
        with tf.GradientTape() as gen_tape:
            fake_images = self.generator(noise, training=True)
            # training=False?
            d_fake = self.discriminator(fake_images, training=True)
            g_loss = self.generator_loss(d_fake)

        gen_grads = gen_tape.gradient(g_loss,
                                      self.generator.trainable_variables)

        self.generator_opt.apply_gradients(zip(gen_grads,
                                               self.generator.
                                               trainable_variables))

        # for tensorboard
        self.g_loss(g_loss)
        self.d_loss(new_d_loss)
        self.w_loss((-1)*(d_loss))  # wasserstein distance

    def train(self, training_data, input_to_GAN, epochs) -> None:
        """
        Training the GAN

        Args:
            training_data (np.array): Actual values for comparison
            input_to_GAN (np.array): Random input values in the shape n x Dims
            epochs (int): number of training epochs
        """
        losses = np.zeros((epochs, 4))

        for epoch in range(epochs):
            noise = input_to_GAN
            real_data = training_data  # X1.astype('int')

            # uncommenting this line means that the noise is not paired with
            # the outputs (probably desirable)
            # noise = np.random.normal(size=[noise.shape[0],noise.shape[1]])

            # shuffle each epoch
            real_data, noise = sklearn.utils.shuffle(real_data, noise)

            xx1 = real_data.reshape(self.batches,
                                    self.batch_size,
                                    self.ndims*self.nsteps)

            inpt1 = noise.reshape(self.batches,
                                  self.batch_size,
                                  self.ndims)

            for i in range(self.batch_size):
                self.train_step(inpt1[i], xx1[i])

            print('epoch:', epoch)
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
                                           input_to_GAN,
                                           training_data,
                                           ndims_latent_input):
        """
        Make and train a model to learn the hypersurfaces of the POD
        coefficients

        Args:
            nPOD (int): Number of POD basis functions
            input_to_GAN (np.ndarray): Data as input to GAN
            training_data (np.ndarray): Training data
            ndims_latent_input (int): Number of dimensions in the latent space

        Returns:
            np.ndarray: array of predictions
        """
        # logs to follow losses on tensorboard
        self.make_logs()

        print('beginning training')
        epochs = 500
        self.train(self, training_data, input_to_GAN, epochs)
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


# if __name__ == "__main__":
# import time
# import tensorflow as tf
# import sklearn.utils
# import sklearn.preprocessing
# import datetime
# import numpy as np

# import os

# gan = GAN
# gan.setup
