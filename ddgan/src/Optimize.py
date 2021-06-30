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
    start_from: int = 100
    nPOD: int = 10
    nLatent: int = nPOD
    npredictions: int = 20  # Number of future steps
    optimizer_epochs: int = 5000

    gan: GAN = None
    eigenvals: np.ndarray = None
    bounds: float = 4.0  # set to inf if not in use

    mse = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(5e-3)

    def mse_loss(self, input, output):

        if tf.math.reduce_max(tf.math.abs(output)) > self.bounds:
            return 1.e5

        elif self.eigenvals is not None:
            print(input.shape)
            print(self.eigenvals.shape)
            return self.mse(
                tf.math.multiply(tf.math.sqrt(self.eigenvals), input),
                tf.math.multiply(tf.math.sqrt(self.eigenvals), output)
                    )
        else:
            return self.mse(input, output)

    @tf.function
    def opt_latent_var(self, latent_var: tf.Variable, output: np.ndarray):
        """
        Main input optimization loop optimizing the latent variable
        based on mse

        Args:
            latent_var (tf.variable): Variable to be optimized
            output (np.ndarray): Actual output

        Returns:
            float: loss variable
            float: norm of the latent variables
        """

        with tf.GradientTape() as tape:
            tape.watch(latent_var)
            r = self.gan.generator(latent_var, training=False)

            loss = self.mse_loss(output,
                                 r[:, :self.gan.ndims*(self.gan.nsteps - 1)]
                                 )

        gradients = tape.gradient(loss, latent_var)
        self.optimizer.apply_gradients(zip([gradients], [latent_var]))

        norm_latent_vars = tf.norm(latent_var)

        # clipping to within 2.3 is equivalent to 98%
        # if norm_latent_vars > 2.3:
        #    latent_var = 2.3 / norm_latent_vars * latent_var
        #    tf.print('clipping to ', tf.norm(latent_var))

        return loss, norm_latent_vars

    def timestep_loop(self,
                      real_output: np.ndarray,
                      prev_latent: np.ndarray,
                      attempts: int):
        """
        Optimizes inputs either from a previous timestep or from
        new randomly initialized inputs

        Args:
            real_output (np.ndarray): Actual values
            prev_latent (np.ndarray): Latent values from previous iteration
            attempts (int): Number of optimization iterations

        Returns:
            np.ndarray: Updated values
            list: Loss values
            np.ndarray: Converged values
            np.ndarray: Initial z values
            list: Norm of latent variables
        """

        loss_list = []
        norm_latent_list = []
        init_latent = prev_latent.numpy()

        for j in range(attempts):
            ip = prev_latent

            for epoch in range(self.optimizer_epochs):
                if epoch % 100 == 0:
                    print('Optimizer epoch: \t', epoch)

                loss, norm_latent = self.opt_latent_var(ip, real_output)
                loss_list.append(loss)
                norm_latent_list.append(norm_latent)

            r = self.gan.generator(ip, training=False)
            loss = self.mse(real_output,
                            r[:, :self.gan.ndims*(self.gan.nsteps - 1)])

            ip_np = ip.numpy()

        return ip, loss_list, ip_np, init_latent, norm_latent_list

    def timesteps(self, initial, inn):
        """
        Outermost loop. Collecting the predicted points and
        iterating through predictions

        Args:
            initial (np.ndarray): Initial value array
            inn (np.ndarray): Gan input array

        Returns:
            np.ndarray: Predicted points
        """
        the_input = tf.convert_to_tensor(inn)
        flds = tf.convert_to_tensor(initial)

        losses_from_opt = []
        norm_latent_list = []
        converged = np.zeros((self.npredictions, self.nLatent))
        latent = np.zeros((self.npredictions, self.nLatent))

        current = tf.Variable(tf.zeros([1, self.gan.ndims]))

        for i in range(self.npredictions):
            print('Time step: \t', i)

            updated, loss_opt, converged[i, :], latent[i, :], norm_latent = \
                self.timestep_loop(the_input, current, 1)
            current = updated

            losses_from_opt.append(loss_opt)
            norm_latent_list.append(norm_latent)

            prediction = self.gan.generator(updated, training=False)

            # Last 4 images become next first 4 images
            next_input = prediction[:, self.gan.ndims:]

            # Last image out of 5 is added to list of compressed vars
            new_result = prediction[:, self.gan.ndims*(self.gan.nsteps - 1):]
            flds = tf.concat([flds, new_result], 0)
            the_input = next_input.numpy()

        np.savetxt('optimised_losses.csv', losses_from_opt, delimiter=',')
        np.savetxt('converged_z_values.csv', converged, delimiter=',')
        np.savetxt('initial_z_values.csv', latent, delimiter=',')
        np.savetxt('norm_latent_vars.csv', norm_latent_list, delimiter=',')

        return flds

    def predict(self, training_data, scaling=None, **kwargs) -> np.ndarray:
        """
        Communicator with the optimization scripts

        Args:
            training_data (np.ndarray): Data used in the training of the GAN
            scaling (sklearn.preprocessing.MinMaxScaler, optional): Scaling
             used to normalize training data. Defaults to None.

        Returns:
            np.ndarray: predictions
        """
        assert self.gan is not None, "Please initialize using an active GAN"

        inn = training_data[self.start_from,
                            :(self.gan.nsteps-1)*self.nPOD
                            ].reshape(1, (self.gan.nsteps - 1) * self.nLatent)

        initial_comp = training_data[self.start_from,
                                     :(self.gan.nsteps-1)*self.nPOD
                                     ].reshape(self.gan.nsteps - 1,
                                               self.nLatent
                                               )

        flds = self.timesteps(initial_comp, inn)
        if scaling is not None:
            flds = scaling.inverse_transform(flds).T

        return flds
