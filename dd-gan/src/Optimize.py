


@tf.function
def opt_latent_var(latent_var, output):   #main input optimization loop, optimizes input1 (a tf.variable) based on mse between known real output and generator output
    #inpt = input1
    #rp = real_outpt
    
    #print('******** in opt_latent_var', latent_var.shape, output.shape) 
    #tf.print('******** input1 before optimisation',input1.read_value())
    
    with tf.GradientTape() as tape:
        tape.watch(latent_var)
        r = generator(latent_var, training=False)  
        loss1 = mse_loss(output, r[:,:nLatent*(nsteps - 1)])

    gradients = tape.gradient(loss1, latent_var)
    optimizer.apply_gradients(zip([gradients], [latent_var]))

    norm_latent_vars = tf.norm(latent_var)

    ## clipping to within 2.3 is equivalent to 98%
    #if norm_latent_vars > 2.3:
    #    latent_var = 2.3 / norm_latent_vars * latent_var 
    #    tf.print('clipping to ', tf.norm(latent_var)) 
    
    #tf.print('******** input1 after optimisation',input1.read_value())
    #tf.print('******** inpt after optimisation  ',inpt.read_value())
    return loss1, norm_latent_vars



def timestep_loop(real_outpt1, previous_latent_vars, attempts): #optimizes inputs - either new randonly initialised inputs, or inputs from previous timestep

    inputs = []
    losses = []

    loss_list = []
    norm_latent_vars_list = []

    initial_latent_variables = previous_latent_vars.numpy()
    
    print('in train all but initial, type previous_latent_vars / op2', type(previous_latent_vars)) 

    for j in range(attempts):

        ip = previous_latent_vars
        
        for epoch in range(nepochs_optimiser):
         
            if epoch%100 == 0:   
                print('******** epoch', epoch)            
            loss1, norm_latent_vars = opt_latent_var(ip, real_outpt1)

            loss_list.append(loss1)
            norm_latent_vars_list.append(norm_latent_vars)

        r = generator(ip, training=False)  
        loss = mse_loss(real_outpt1, r[:,:nLatent*(nsteps - 1)])

        inputt = ip.numpy()
        loss_input = loss.numpy()

        #inputs.append(inputt)
        #losses.append(loss_input)


    #initial_inputs = np.array(inputs)
    #loss_for_initial_inputs = np.array(losses)
    #initial_inputs = inputt
    #loss_for_initial_inputs = loss_input

    #min_loss = np.argmin(loss_for_initial_inputs)
    #best_ipt = initial_inputs[min_loss]

    return ip, loss_list, inputt, initial_latent_variables, norm_latent_vars_list #best_ipt


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
