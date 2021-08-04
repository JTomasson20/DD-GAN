import tensorflow as tf
import numpy as np
from .Train import GAN
from dataclasses import dataclass

__all__ = ['Optimize']


@dataclass(unsafe_hash=True)
class Optimize:
    """
    Finding position and orienting within the latent space
    to predict in time.
    """

    # Input data hyperparameters
    start_from: int = 100
    nPOD: int = 10
    nLatent: int = 10  # Dimensionality of latent space
    dt: int = 1

    # Optimization hyperparameters
    npredictions: int = 20  # Number of future steps
    optimizer_epochs: int = 5000
    attempts = 1

    # Only for DD
    evaluated_subdomains: int = 2
    cycles: int = 3
    cumulative_steps = None
    dim_steps = None

    # Objects
    mse = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(5e-3)
    gan: GAN = None

    # Number of optimized points, would normally keep
    # this as a private variable
    nOptimized: int = 90

    # Debug mode
    debug: bool = False

    # Zeros, Past or None
    initial_values: str = "Past"
    reset: bool = False
    disturb: bool = False

    def mse_loss(self, input, output):
        """Mean square error loss function

        Args:
            input (tensor): Predicted values
            output (tensor): Actual value

        Returns:
            int: mse loss value
        """

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
                        prev_latent: np.ndarray
                        ) -> np.ndarray:
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

    def timesteps(self,
                  initial: np.ndarray,
                  inn: np.ndarray
                  ) -> np.ndarray:
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

            # Only one step at a time is changed so start by overwriting
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

    def predict(self,
                training_data: np.ndarray,
                scaling=None
                ) -> np.ndarray:
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

        flds = self.timesteps(initial_comp, inn)

        if scaling is not None:
            flds = scaling.inverse_transform(flds).T

        return flds

    def timestepsDD(self,
                    initial: np.ndarray,
                    inn: np.ndarray,
                    boundrary_condition: np.ndarray
                    ) -> list:
        """
        Outermost loop. Collecting the predicted points and
        iterating through predictions

        Args:
            initial (np.ndarray): Initial value array
            inn (np.ndarray): Gan input array
            boundrary_condition (np.ndarray): values for
                leftmost and rightmost domains
        Returns:
            np.ndarray: Predicted points
        """
        # Dynamic predictions. Always same shape
        the_input = tf.convert_to_tensor(inn)

        # Structured predictions. Data added as we go
        flds = tf.convert_to_tensor(
            np.array(initial)[:, 1-self.dim_steps[-1]:]
            )

        # Indexes for the next iteration
        indexing = np.arange(self.nPOD, self.nPOD + self.nOptimized)

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
             self.nPOD + self.nOptimized))

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
            if self.debug:
                print(np.array(the_input))

            the_input = self.add_bc(the_input, i, boundrary_condition)

            if self.debug:
                print(np.array(the_input))

            # Do a predict/update cycle
            for _ in range(self.cycles):
                print("Cycle: \t " + str(_))
                for j in domains:
                    print("Domain: \t " + str(j))

                    if self.disturb:
                        current_latent_single.assign(
                            current_latent[j] +
                            tf.random.normal(
                                [1, self.nLatent],
                                mean=0.0,
                                stddev=0.05)
                                )
                    else:
                        current_latent_single.assign(current_latent[j])

                    tmp[0] = self.timestep_loopDD(
                        the_input[j],
                        current_latent_single)

                    # Essentially updated[j] = tmp
                    updated = self.update_updated(tmp, updated, j)

                    # Update the next prediction
                    prediction[j, 0] = self.gan.generator(
                        updated[j],
                        training=False)

                    the_input = self.communicate(the_input,
                                                 prediction,
                                                 j, domains)

                    current_latent = updated
            # Redo the first one to complete the final cycle
            print("Domain: \t 0")
            if self.disturb:
                current_latent_single.assign(
                    current_latent[0] +
                    tf.random.normal(
                        [1, self.nLatent],
                        mean=0.0,
                        stddev=0.05)
                        )
            else:
                current_latent_single.assign(current_latent[0])

            tmp[0] = self.timestep_loopDD(
                    the_input[0],
                    current_latent_single)

            # updated[j] = tmp
            updated = self.update_updated(tmp, updated, 0)

            prediction[0, 0] = self.gan.generator(
                updated[0], training=False)

            the_input = self.communicate(the_input,
                                         prediction,
                                         0, domains)
            current_latent = updated
            if self.debug:
                print(np.array(the_input))

            the_input = self.initial_guess(the_input)

            if self.debug:
                print(np.array(the_input))

            # Last 4 images become next first 4 images
            prediction[:, :, :self.nOptimized] = the_input
            next_input = prediction[:, :, indexing]

            # Last image out of 5 is added to list of compressed vars
            new_result = prediction[:, :, self.nOptimized:]
            flds = tf.concat([flds, new_result], 1)
            the_input = tf.convert_to_tensor(next_input, dtype=np.float32)
            if self.debug:
                print(np.array(the_input))

        return flds

    def predictDD(self,
                  training_data: np.ndarray,
                  boundrary_conditions: np.ndarray,
                  dim_steps=None
                  ) -> list:
        """
        Prediction script if Domain Decomposition is applied

        Args:
            training_data (np.ndarray)
            boundrary_condition (np.ndarray): values for
                leftmost and rightmost domains
            dim_steps (np.ndarray): number of samples in each
                dimension of gan
        """
        # Updating these values only if DD is used
        self.evaluated_subdomains = len(training_data)
        if dim_steps is None:
            self.dim_steps = np.ones(3)*self.gan.nsteps
            self.nOptimized = (self.gan.nsteps*3 - 1) * self.nPOD
        else:
            self.dim_steps = dim_steps
            self.nOptimized = (np.sum(dim_steps)-1) * self.nPOD

        self.cumulative_steps = np.cumsum(self.dim_steps)
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

    def initial_guess(self,
                      the_input,
                      ):
        """
        Adding initial guesses for next iteration

        Args:
            the_input (tensor): Current iteration tensor

        Returns:
            tensor: Updated iteration tensor
        """
        if self.initial_values == "Zeros":
            for k in range(len(the_input)):
                for j in range(2):
                    the_input = tf.tensor_scatter_nd_update(
                        the_input,
                        np.array(
                            [np.ones(self.nPOD)*k,
                             np.zeros(self.nPOD),
                             np.arange(self.nPOD*(self.cumulative_steps[j]),
                                       self.nPOD*(self.cumulative_steps[j]+1)
                                       )],
                            dtype=np.int32).T,
                        np.zeros(self.nPOD))

        elif self.initial_values == "Past":
            for k in range(len(the_input)):
                for j in range(2):
                    the_input = tf.tensor_scatter_nd_update(
                        the_input,
                        np.array(
                            [np.ones(self.nPOD)*k,
                             np.zeros(self.nPOD),
                             np.arange(self.nPOD*(self.cumulative_steps[1-j]),
                                       self.nPOD*(self.cumulative_steps[1-j]+1)
                                       )],
                            dtype=np.int32).T,
                        the_input[k, 0,
                                  self.nPOD*(self.cumulative_steps[1-j]-1):
                                  self.nPOD*(self.cumulative_steps[1-j])])

        return the_input

    def add_bc(self,
               the_input,
               i: int,
               boundrary_condition: np.ndarray
               ):
        """
        Adding boundrary conditions to left and rightmost domains

        Args:
            the_input (tensor): Current iteration tensor
            i (int): Iteration number
            boundrary_condition (np.ndarray): Boundrary values

        Returns:
            tensor: Updated iteration tensor
        """
        # Doing boundrary conditions
        # Input the correct data to left boundrary
        the_input = tf.tensor_scatter_nd_update(
            the_input,
            np.array(
                [np.zeros(self.nPOD),
                 np.zeros(self.nPOD),
                 np.arange(self.nPOD*(self.cumulative_steps[0] - 1),
                           self.nPOD*self.cumulative_steps[0])],
                dtype=np.int32).T,
            boundrary_condition[
                0, i*self.dt + self.start_from])

        # Input data to right boundrary
        the_input = tf.tensor_scatter_nd_update(
            the_input,
            np.array(
                [np.ones(self.nPOD) * (len(the_input)-1),
                    np.zeros(self.nPOD),
                    np.arange(self.nPOD*(self.cumulative_steps[1] - 1),
                              self.nPOD*self.cumulative_steps[1])],
                dtype=np.int32).T,
            boundrary_condition[
                1, i*self.dt + self.start_from])
        # Initializing predicted values

        return the_input

    def communicate(self,
                    the_input,
                    prediction: np.ndarray,
                    j: int,
                    domains: np.ndarray
                    ):
        """
        Communicate with neighbouring subdomains

        Args:
            the_input (tensor): Current iteration guess
            prediction (np.ndarray): Latent values
            j (int): Domain number
            domains (np.ndarray): Iteration ordering of domains

        Returns:
            tensor: Updated iteration tensor
        """
        # Communicate the update to neighbours
        # everybody except leftmost updates to the left
        if j != 0:
            the_input = tf.tensor_scatter_nd_update(
                the_input,
                np.array(
                    [np.ones(self.nPOD) * (j-1),
                        np.zeros(self.nPOD),
                        np.arange(
                            self.nPOD*(self.cumulative_steps[1]-1),
                            self.nPOD*self.cumulative_steps[1])],
                    dtype=np.int32).T,
                prediction[j, 0, -self.nPOD:])

        # everybody except rightmost updates to the right
        if j != np.max(domains):
            the_input = tf.tensor_scatter_nd_update(
                the_input,
                np.array(
                    [np.ones(self.nPOD) * (j+1),
                        np.zeros(self.nPOD),
                        np.arange(
                            self.nPOD*(self.cumulative_steps[0] - 1),
                            self.nPOD*self.cumulative_steps[0])],
                    dtype=np.int32).T,
                prediction[j, 0, -self.nPOD:])

        return the_input

    def update_updated(self,
                       tmp: np.ndarray,
                       updated,
                       j: int
                       ):
        """Update latent values

        Args:
            tmp (np.ndarray): Values for current timestep
            updated (tensor): Values for previous timestep
            j (int): Domain number

        Returns:
            tensor: Updated tensor
        """
        updated = tf.tensor_scatter_nd_update(
                        updated,
                        np.array([
                            np.ones(self.nLatent)*j,
                            np.zeros(self.nLatent),
                            np.arange(self.nLatent)],
                         dtype=np.int32).T,
                        tmp[0][0])
        return updated
