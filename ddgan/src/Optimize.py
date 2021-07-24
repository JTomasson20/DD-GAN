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
    nLatent: int = 10  # Dimensionality of latent space
    npredictions: int = 20  # Number of future steps
    optimizer_epochs: int = 5000
    attempts = 1

    gan: GAN = None
    eigenvals: np.ndarray = None

    # Only for DD
    evaluated_subdomains: int = 2
    cycles: int = 3

    mse = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(5e-3)

    # Number of optimized points, would normally keep
    # this as a private variable
    nOptimized: int = 90

    # Set to false for previous version of updating
    # predictions
    mod: bool = True

    def mse_loss(self, input, output):

        # if self.eigenvals is not None:
        #     print(input.shape)
        #     print(self.eigenvals.shape)
        #     return self.mse(
        #         tf.math.multiply(tf.math.sqrt(self.eigenvals), input),
        #         tf.math.multiply(tf.math.sqrt(self.eigenvals), output)
        #             )
        # else:
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
                                 r[:, :self.nOptimized]
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
                      prev_latent: np.ndarray):
        """
        Optimizes inputs either from a previous timestep or from
        new randomly initialized inputs

        Args:
            real_output (np.ndarray): Actual values
            prev_latent (np.ndarray): Latent values from previous iteration
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

        for j in range(self.attempts):
            ip = prev_latent

            for epoch in range(self.optimizer_epochs):
                if epoch % 100 == 0:
                    print('Optimizer epoch: \t', epoch)

                loss, norm_latent = self.opt_latent_var(ip, real_output)
                loss_list.append(loss)
                norm_latent_list.append(norm_latent)

            r = self.gan.generator(ip, training=False)
            loss = self.mse(real_output,
                            r[:, :self.nOptimized])

            ip_np = ip.numpy()

        return ip, loss_list, ip_np, init_latent, norm_latent_list

    def timestep_loopDD(self,
                        real_output: np.ndarray,
                        prev_latent: np.ndarray):
        """
        Optimizes inputs either from a previous timestep or from
        new randomly initialized inputs

        Args:
            real_output (np.ndarray): Actual values
            prev_latent (np.ndarray): Latent values from previous iteration
        Returns:
            np.ndarray: Updated values

        """

        for j in range(self.attempts):
            ip = prev_latent
            for epoch in range(self.optimizer_epochs):
                if epoch % 100 == 0:
                    print('Optimizer epoch: \t', epoch)

                __, ___ = self.opt_latent_var(ip, real_output)

        return ip

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

        current = tf.Variable(tf.zeros([1, self.gan.nsteps]))
        # prediction = np.zeros((1, self.nLatent * self.nPOD))

        for i in range(self.npredictions):
            print('Time step: \t', i)

            updated, loss_opt, converged[i, :], latent[i, :], norm_latent = \
                self.timestep_loop(the_input, current)
            current = updated

            losses_from_opt.append(loss_opt)
            norm_latent_list.append(norm_latent)

            prediction = self.gan.generator(updated, training=False)

            # Last 4 images become next first 4 images
            next_input = prediction[:, self.gan.ndims:]

            # Last image out of 5 is added to list of compressed vars
            new_result = prediction[:, self.nOptimized:]
            flds = tf.concat([flds, new_result], 0)
            the_input = next_input.numpy()

        np.savetxt('optimised_losses.csv', losses_from_opt, delimiter=',')
        np.savetxt('converged_z_values.csv', converged, delimiter=',')
        np.savetxt('initial_z_values.csv', latent, delimiter=',')
        np.savetxt('norm_latent_vars.csv', norm_latent_list, delimiter=',')

        return flds

    def timesteps_mod(self, initial, inn):
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

        current = tf.Variable(tf.zeros([1, self.gan.nsteps]))
        prediction = np.zeros((1, self.nLatent * self.nPOD))

        for i in range(self.npredictions):
            print('Time step: \t', i)

            updated, loss_opt, converged[i, :], latent[i, :], norm_latent = \
                self.timestep_loop(the_input, current)
            current = updated

            losses_from_opt.append(loss_opt)
            norm_latent_list.append(norm_latent)

            prediction[0] = self.gan.generator(updated, training=False)

            # Not so sure about the this line but it makes sense to me
            # that only one step at a time is changed so start by overwriting
            # the first steps with the previous values
            prediction[:, :self.nOptimized] = the_input

            # Last 4 images become next first 4 images
            next_input = prediction[:, self.gan.ndims:]

            # Last image out of 5 is added to list of compressed vars
            new_result = prediction[:, self.nOptimized:]
            flds = tf.concat([flds, new_result], 0)
            the_input = tf.convert_to_tensor(next_input)

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
        self.nOptimized = self.gan.ndims*(self.gan.nsteps - 1)

        inn = training_data[
            self.start_from,
            :(self.gan.nsteps-1)*self.nPOD
            ].reshape(1, self.nOptimized)

        initial_comp = training_data[
            self.start_from,
            :(self.gan.nsteps-1)*self.nPOD
            ].reshape(self.gan.nsteps - 1, self.nPOD)

        if self.mod:
            flds = self.timesteps_mod(initial_comp, inn)
        else:
            flds = self.timesteps(initial_comp, inn)

        if scaling is not None:
            flds = scaling.inverse_transform(flds).T

        return flds

    def timestepsDD(self, initial, inn, boundrary_condition):
        """
        Outermost loop. Collecting the predicted points and
        iterating through predictions

        Args:
            initial (np.ndarray): Initial value array
            inn (np.ndarray): Gan input array

        Returns:
            np.ndarray: Predicted points
        """
        # Dynamic predictions. Always same shape
        the_input = tf.convert_to_tensor(inn)

        # Structured predictions. Data added as we go
        flds = tf.convert_to_tensor(np.array(initial)[:, 1-self.gan.nsteps:])

        # Indexes for the next iteration
        indexing = np.arange(self.nPOD, self.nPOD * self.gan.nsteps * 3)

        # Latent space
        current_latent = tf.Variable(tf.zeros([
            self.evaluated_subdomains,
            1,
            self.nLatent
            ]))

        current_latent_single = tf.Variable(tf.zeros([
            1,
            self.nLatent
            ]))

        updated = current_latent

        # Predicted POD coefficients
        prediction = np.zeros(
            (self.evaluated_subdomains,
             1,
             self.nLatent * self.nPOD * 3))

        # Moving back and forward to be iterated through
        # 2 domains become (0,1)
        # 3 domains become (0,1,2,1)
        domains = np.concatenate((
            np.arange(self.evaluated_subdomains),
            np.arange(self.evaluated_subdomains)[1:-1][::-1]
        ))

        tmp = [0]

        for i in range(self.npredictions):
            print('Time step: \t', i)

            # Doing boundrary conditions
            # Input the correct data to left boundrary
            the_input = tf.tensor_scatter_nd_update(
                the_input,
                np.array(
                    [np.zeros(self.nPOD),
                     np.zeros(self.nPOD),
                     np.arange(self.nPOD*(self.gan.nsteps - 1),
                               self.nPOD*self.gan.nsteps)],
                    dtype=np.int32).T,
                boundrary_condition[
                    0, i + self.start_from])

            # Input data to right boundrary
            the_input = tf.tensor_scatter_nd_update(
                the_input,
                np.array(
                    [np.ones(self.nPOD) * (len(the_input)-1),
                     np.zeros(self.nPOD),
                     np.arange(self.nPOD*(2*self.gan.nsteps - 1),
                               self.nPOD*2*self.gan.nsteps)],
                    dtype=np.int32).T,
                boundrary_condition[
                    1, i + self.start_from])

            # Do a predict/update cycle
            for _ in range(self.cycles):
                print("Cycle: \t " + str(_))
                for j in domains:
                    print("Domain: \t " + str(j))
                    current_latent_single.assign(current_latent[j])
                    tmp[0] = self.timestep_loopDD(
                        the_input[j],
                        current_latent_single)

                    # Essentially updated[j] = tmp
                    updated = tf.tensor_scatter_nd_update(
                        updated,
                        np.array([
                            np.ones(self.nLatent)*j,
                            np.zeros(self.nLatent),
                            np.arange(self.nLatent)],
                         dtype=np.int32).T,
                        tmp[0][0])

                    # Update the next prediction
                    prediction[j, 0] = self.gan.generator(
                        updated[j],
                        training=False)

                    # Communicate the update to neighbours
                    # everybody except leftmost updates to the left
                    if j != 0:
                        the_input = tf.tensor_scatter_nd_update(
                            the_input,
                            np.array(
                                [np.ones(self.nPOD) * (j-1),
                                 np.zeros(self.nPOD),
                                 np.arange(self.nPOD*(2*self.gan.nsteps - 1),
                                           self.nPOD*2*self.gan.nsteps)],
                                dtype=np.int32).T,
                            prediction[j, 0, -self.nPOD:])

                    # everybody except rightmost updates to the right
                    if j+1 != self.evaluated_subdomains:
                        the_input = tf.tensor_scatter_nd_update(
                            the_input,
                            np.array(
                                [np.ones(self.nPOD) * (j+1),
                                 np.zeros(self.nPOD),
                                 np.arange(self.nPOD*(self.gan.nsteps - 1),
                                           self.nPOD*self.gan.nsteps)],
                                dtype=np.int32).T,
                            prediction[j, 0, -self.nPOD:])

                    current_latent = updated

                # Redo the first one to complete the final cycle
                current_latent_single.assign(current_latent[0])
                tmp[0] = self.timestep_loopDD(
                        the_input[j],
                        current_latent_single)

                # updated[j] = tmp
                updated = tf.tensor_scatter_nd_update(
                    updated,
                    np.array([
                        np.zeros(self.nLatent),
                        np.zeros(self.nLatent),
                        np.arange(self.nLatent)],
                        dtype=np.int32).T,
                    tmp[0][0])

                prediction[0, 0] = self.gan.generator(
                    updated[0], training=False)

                current_latent = updated

            # Last 4 images become next first 4 images
            prediction[:, :, :self.nOptimized] = the_input
            next_input = prediction[:, :, indexing]

            # Last image out of 5 is added to list of compressed vars
            new_result = prediction[:, :, self.nOptimized:]
            flds = tf.concat([flds, new_result], 1)
            the_input = tf.convert_to_tensor(next_input, dtype=np.float32)

        return flds

    def predictDD(self, training_data, boundrary_conditions):
        """
        Communicator with optimizaiton scripts.
        Breaks up training data and feeds it forward.

        Args:
            training_data (np.ndarray)
        """
        # Updating these values only if DD is used
        self.evaluated_subdomains = len(training_data)
        self.nOptimized = (self.gan.nsteps*3 - 1) * self.nPOD

        inn = []  # Flattened version of data
        initial_comp = []  # Structured version of data

        for i in range(self.evaluated_subdomains):
            inn.append(training_data[
                i, self.start_from, :self.nOptimized
                ].reshape(1, self.nOptimized))

            initial_comp.append(training_data[
                i, self.start_from, :self.nOptimized
                ].reshape(-1, self.nPOD))

        return self.timestepsDD(initial_comp, inn, boundrary_conditions)
