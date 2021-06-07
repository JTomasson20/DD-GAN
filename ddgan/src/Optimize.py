import tensorflow as tf
from src.Train import GAN

__all__ = []

def mse_loss(inp, outp):
    return tf.keras.losses.MeanSquaredError(inp, outp)

@tf.function
def opt_latent_var(gan, latent_var, output):
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
    optimizer = tf.keras.optimizers.Adam(5e-3)
    
    with tf.GradientTape() as tape:
        tape.watch(latent_var)
        r = gan.generator(latent_var, training=False)  
        loss1 = mse_loss(output, r[:, :gan.ndims*(gan.nsteps - 1)])

    gradients = tape.gradient(loss1, latent_var)
    optimizer.apply_gradients(zip([gradients], [latent_var]))

    norm_latent_vars = tf.norm(latent_var)

    # clipping to within 2.3 is equivalent to 98%
    # if norm_latent_vars > 2.3:
    #    latent_var = 2.3 / norm_latent_vars * latent_var
    #    tf.print('clipping to ', tf.norm(latent_var))

    return loss1, norm_latent_vars


def timestep_loop(gan,
                  real_outpt1,
                  previous_latent_vars,
                  attempts,
                  optimizer_epochs=5000):
    # What kind of an otherwordly mess is this function
    """
    Optimizes inputs either from a previous timestep or from
    new randomly initialized inputs
    """
    inputs = []
    losses = []

    loss_list = []
    norm_latent_vars_list = []

    initial_latent_variables = previous_latent_vars.numpy()

    for j in range(attempts):
        ip = previous_latent_vars
        for epoch in range(optimizer_epochs):
            if epoch % 100 == 0:
                print('Optimizer epoch: ', epoch)
            loss1, norm_latent_vars = opt_latent_var(ip, real_outpt1)

            loss_list.append(loss1)
            norm_latent_vars_list.append(norm_latent_vars)

        r = gan.generator(ip, training=False)  
        loss = mse_loss(real_outpt1, r[:, :nLatent*(nsteps - 1)])

        inputt = ip.numpy()
        loss_input = loss.numpy()

        inputs.append(inputt)
        losses.append(loss_input)


    # initial_inputs = np.array(inputs)
    # loss_for_initial_inputs = np.array(losses)
    # initial_inputs = inputt
    # loss_for_initial_inputs = loss_input

    # min_loss = np.argmin(loss_for_initial_inputs)
    # best_ipt = initial_inputs[min_loss]

    return ip, loss_list, inputt, initial_latent_variables, norm_latent_vars_list


def timesteps(initial, inn, iterations):  #timestep prediction
    next_input1 = tf.convert_to_tensor(inn)
    flds = tf.convert_to_tensor(initial)

    losses_from_opt = []
    norm_latent_vars_all_time_list = []
    converged_inputs = np.zeros((iterations, 5))
    initial_latent =  np.zeros((iterations, 5))
  
    ip1 = tf.zeros([1, nLatent]) #tf.random.normal([1, nLatent])
    current = tf.Variable(ip1)

    for i in range(iterations):
        print ('*** predicting time step ',i)
        
        # attempts=1 hard-wired
        updated, loss_opt, converged_inputs[i,:], initial_latent[i,:], norm_latent_vars_list = timestep_loop(next_input1, current, 1) 
        current = updated

        losses_from_opt.append(loss_opt)
        norm_latent_vars_all_time_list.append(norm_latent_vars_list)
        #print('norm_latent_vars_list:', len(norm_latent_vars_list), type(norm_latent_vars_list))

        prediction = generator(updated, training=False)
        #print('*** evaluate the generator with op2', prediction.numpy())
        next_input = prediction[:,nLatent:] #last 4 images become next first 4 images
        
        new_result = prediction[:,nLatent*(nsteps - 1):]    #last image out of 5 is added to list of compressed vars
        flds = tf.concat([flds, new_result], 0)

        next_input1 = next_input.numpy()

    #print('types loss_opt and norm_latent_vars', type(losses_from_opt), type(norm_latent_vars_all_time_list))

    #np.savetxt('final_5_time_levels.csv', r_values, delimiter=',')
    np.savetxt('optimised_losses.csv', losses_from_opt, delimiter=',')
    np.savetxt('converged_z_values.csv', converged_inputs, delimiter=',')
    np.savetxt('initial_z_values.csv', initial_latent, delimiter=',')
    np.savetxt('norm_latent_vars.csv',norm_latent_vars_all_time_list,delimiter=',')

    return flds
