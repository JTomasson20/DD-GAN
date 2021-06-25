import tensorflow as tf
import numpy as np
from .Train import GAN
from dataclasses import dataclass

__all__ = ['Optimize']


@dataclass(unsafe_hash=True)
class Optimize:
    """
    Finding and orienting within the latent space. Functions are
    in the order innermost -> outermost. The function therefore most
    often called appears first
    """
    start_from: int = 0
    nPOD: int = 15
    iterations: int = 10
    npredictions: int = 20
    optimizer_epochs: int = 5000
    gan: GAN = None

    latent_size: int = 100
    ntimes: int = 20

    mse = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(1e-2)

    def opt_step(self, latent_values, real_coding):
        with tf.GradientTape() as tape:
            tape.watch(latent_values)
            gen_output = self.gan.generator(latent_values, training=False)
            loss = self.mse(
                real_coding,
                gen_output[:, :(self.ntimes - 1), :, :]
                )

        gradient = tape.gradient(loss, latent_values)
        self.optimizer.apply_gradients(zip([gradient], [latent_values]))

        return loss

    def optimize_coding(self, latent_values, real_coding):

        for epoch in range(self.optimizer_epochs):
            self.opt_step(latent_values, real_coding)

        return latent_values

    def predict(self, X_train_concat, scaling=None, **kwargs) -> np.ndarray:
        """
        Communicator with the optimization scripts

        Args:
            training_data (np.ndarray): Data used in the training of the GAN
            scaling (sklearn.preprocessing.MinMaxScaler, optional): Scaling
             used to normalize training data. Defaults to None.

        Returns:
            np.ndarray: predictions
        """

        real_coding = X_train_concat[0].reshape(1, -1)
        real_coding = real_coding[:, : self.nPOD*(self.ntimes - 1)]
        real_coding = tf.constant(real_coding)
        real_coding = tf.cast(real_coding, dtype=tf.float32)

        latent_values = tf.random.normal([len(real_coding), self.latent_size])
        latent_values = tf.Variable(latent_values)

        latent_values = self.optimize_coding(latent_values, real_coding)

        X_predict = list(
            self.gan.generator(latent_values).numpy().reshape(-1, self.nPOD)
            )
        gen_predict = X_predict[-1]
        real_coding = np.concatenate((
            real_coding,
            gen_predict.reshape(1, -1)), axis=1
            )[:, self.nPOD:]
        real_coding = tf.constant(real_coding)
        real_coding = tf.cast(real_coding, dtype=tf.float32)

        for i in range(self.npredictions):
            print("Prediction: ", i, " / ", self.npredictions)
            latent_values = self.optimize_coding(latent_values, real_coding)
            gen_predict = self.gan.generator(
                latent_values
                )[:, (self.ntimes - 1):, :, :].numpy()
            X_predict.append(gen_predict.flatten())
            real_coding = np.concatenate((
                real_coding,
                gen_predict.reshape(1, -1)), axis=1
                )[:, self.nPOD:]
            real_coding = tf.constant(real_coding)
            real_coding = tf.cast(real_coding, dtype=tf.float32)
        X_predict = np.array(X_predict)

        return X_predict
