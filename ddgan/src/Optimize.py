import tensorflow as tf
import numpy as np
from src.Train import GAN

__all__ = []

from dataclasses import dataclass


@dataclass
class Optimize:
    """
    Finding and orienting within the latent space
    """
    initial: int = None
    inn: int = None
    iterations: int = None
    optimizer_epochs: int = None
    gan: GAN = gan

    optimizer = tf.keras.optimizers.Adam(5e-3)

    def mse_loss(inp, outp):
        return tf.keras.losses.MeanSquaredError(inp, outp)

    @tf.function
    def opt_latent_var(self, latent_var, output):
        """
        Main input optimization loop optimizing the latent variable
        based on mse

        Args:
            gan (GAN object) : Generator-discriminator pair
            latent_var (tf.variable): Variable to be optimized
            output (np.array): Actual output

        Returns:
            float: loss variable
            float: norm of the latent variables
        """
        
        with tf.GradientTape() as tape:
            tape.watch(latent_var)
            r = self.gan.generator(latent_var, training=False)
            loss = self.mse_loss(output,
                                 r[:, :self.gan.ndims*(self.gan.nsteps - 1)])

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
        # What kind of an otherwordly mess is this function
        """
        Optimizes inputs either from a previous timestep or from
        new randomly initialized inputs
        """
        inputs = []
        losses = []

        loss_list = []
        norm_latent_list = []

        init_latent = prev_latent.numpy()

        for j in range(attempts):
            ip = previous_latent_vars
            for epoch in range(self.optimizer_epochs):
                if epoch % 100 == 0:
                    print('Optimizer epoch: ', epoch)
                
                loss, norm_latent = self.opt_latent_var(ip, real_output)
                loss_list.append(loss)
                norm_latent_list.append(norm_latent)

            r = self.gan.generator(ip, training=False)  
            loss = self.mse_loss(real_output,
                                 r[:, :self.gan.ndims*(self.gan.nsteps - 1)])

            ip_np = ip.numpy()

            inputs.append(ip_np)
            losses.append(loss.numpy())

        return ip, loss_list, ip_np, init_latent, norm_latent_list


    def timesteps(initial, inn, iterations):
        """
        Timestep prediction
        """
        the_input = tf.convert_to_tensor(inn)
        flds = tf.convert_to_tensor(initial)

        losses_from_opt = []
        norm_latent_vars_all_time_list = []
        converged_inputs = np.zeros((iterations, 5))
        initial_latent = np.zeros((iterations, 5))
    
        ip1 = tf.zeros([1, nLatent]) #tf.random.normal([1, nLatent])
        current = tf.Variable(ip1)

        for i in range(iterations):
            print ('Time step \t', i)
            
            updated, loss_opt, converged_inputs[i,:], initial_latent[i,:], norm_latent_vars_list = timestep_loop(the_input, current, 1) 
            current = updated

            losses_from_opt.append(loss_opt)
            norm_latent_vars_all_time_list.append(norm_latent_vars_list)

            prediction = self.gan.generator(updated, training=False)
            next_input = prediction[:,nLatent:] #last 4 images become next first 4 images
            
            new_result = prediction[:,nLatent*(nsteps - 1):]    #last image out of 5 is added to list of compressed vars
            flds = tf.concat([flds, new_result], 0)

            the_input = next_input.numpy()

        #print('types loss_opt and norm_latent_vars', type(losses_from_opt), type(norm_latent_vars_all_time_list))

        #np.savetxt('final_5_time_levels.csv', r_values, delimiter=',')
        np.savetxt('optimised_losses.csv', losses_from_opt, delimiter=',')
        np.savetxt('converged_z_values.csv', converged_inputs, delimiter=',')
        np.savetxt('initial_z_values.csv', initial_latent, delimiter=',')
        np.savetxt('norm_latent_vars.csv',norm_latent_vars_all_time_list,delimiter=',')

        return flds
