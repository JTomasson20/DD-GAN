import time
import tensorflow as tf
import sklearn.utils
import sklearn.preprocessing
import datetime
import numpy as np

import os

def make_generator(nsteps):
    generator = tf.keras.Sequential()
    generator.add(tf.keras.layers.Dense(5, input_shape=(5,), activation='relu')) # 5
    generator.add(tf.keras.layers.BatchNormalization())
    generator.add(tf.keras.layers.Dense(10, activation='relu'))                  # 10
    generator.add(tf.keras.layers.BatchNormalization())
    generator.add(tf.keras.layers.Dense((5*nsteps), activation='relu'))          # 25
    generator.add(tf.keras.layers.BatchNormalization())
    generator.add(tf.keras.layers.Dense((5*nsteps), activation='tanh'))          # 25

    return generator


def make_critic(nsteps):
    discriminator = tf.keras.Sequential()
    discriminator.add(tf.keras.layers.Dense(5*nsteps, input_shape=(5*nsteps,)))
    discriminator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    discriminator.add(tf.keras.layers.Dropout(0.3))
    discriminator.add(tf.keras.layers.Dense(10))
    discriminator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    discriminator.add(tf.keras.layers.Dense(5))
    discriminator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    discriminator.add(tf.keras.layers.Dropout(0.3))
    discriminator.add(tf.keras.layers.Flatten())
    discriminator.add(tf.keras.layers.Dense(1))


    return discriminator


def discriminator_loss(d_real, d_fake):
    d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
    return d_loss

def generator_loss(d_fake):
    g_loss = -tf.reduce_mean(d_fake)
    return g_loss

@tf.function
def train_step(noise, real, lmbda, n_critic, g, c, g_opt, c_opt, g1_loss, d1_loss, w_loss):   
    batch_size = len(real)

    ##### train critic ######
    
    #print('inside train step') 

    for i in range(n_critic):    
        with tf.GradientTape() as t:
            with tf.GradientTape() as t1:
                fake = g(noise, training=True) # training=False?
                epsilon = tf.random.uniform(shape=[batch_size, 1], minval=0., maxval=1.)
                #print ('types', real.dtype, epsilon.dtype, fake.dtype)                
                interpolated = real + epsilon * (fake - real)  
                t1.watch(interpolated)
                c_inter = c(interpolated, training=True)  

                d_real = c(real, training=True)
                d_fake = c(fake, training=True)
                d_loss = discriminator_loss(d_real, d_fake)   #initial c loss
                
                #print('d loop', d_real.shape,  d_fake.shape,  d_loss.shape, c_inter.shape) # batch_size by 1 except for d_loss
                #print('d loop',  interpolated.shape, inpt.shape, real.shape, fake.shape) # 
                
                
                #print('discriminator loop',i ,'---------------------------------------------')    
                #print('min and max values of fake',np.min(fake.numpy()),np.max(fake.numpy()))
                #print('min and max values of real',np.min(real),np.max(real) )
                #print('min and max values of d_fake',np.min(d_fake.numpy()),np.max(d_fake.numpy()))
                #print('min and max values of d_real',np.min(d_real.numpy()),np.max(d_real.numpy()))
                
            grad_interpolated = t1.gradient(c_inter, interpolated)
            
            #print('interpolated      ', interpolated.numpy())
            #print('c_inter           ', c_inter.numpy())
            #print('grad_interpolated', grad_interpolated.numpy())
            
            #print('grad_interpolated itself ', grad_interpolated.numpy().shape) # batch_size by 25
            #print('grad_interpolated square ', tf.square(grad_interpolated).numpy().shape) # batch_size by 25
            #print('grad_interpolated red sum', tf.reduce_sum(tf.square(grad_interpolated), axis=[1]).numpy().shape) # batch_size by 1
            #print('grad_interpolated sqrt', tf.sqrt(tf.reduce_sum(tf.square(grad_interpolated), axis=[1])).numpy().shape) # batch_size by 1            
            
                     #tf.sqrt(tf.reduce_sum(tf.square(x)) + 1.0e-12)
            slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_interpolated) + 1e-12, axis=[1])) # 
            
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            #print('slopes, grad penalty', slopes.numpy(), gradient_penalty.numpy())

            new_d_loss = d_loss + (lmbda*gradient_penalty)  #new c loss
            #print('d_loss and new_d_loss',d_loss.numpy(), new_d_loss.numpy()) 
        
        c_grad = t.gradient(new_d_loss, c.trainable_variables)
        #print('length and type c_grad', len(c_grad), type(c_grad))  # length of c_grad was 8? type list
        #print('length and type c_grad[0]', (c_grad[0].shape), type(c_grad[0]))  # length of c_grad was 8? type list
        #print('values of cgrad',c_grad)
        #print('c.trainable_variables', c.trainable_variables)
        c_opt.apply_gradients(zip(c_grad, c.trainable_variables))


    ##### train generator #####

    with tf.GradientTape() as gen_tape:
        fake_images = g(noise, training=True)
        d_fake = c(fake_images, training=True) # training=False?
        g_loss = generator_loss(d_fake)

    gen_grads = gen_tape.gradient(g_loss, g.trainable_variables)
    g_opt.apply_gradients(zip(gen_grads, g.trainable_variables))
    
    
    #print('g.trainable_variables')
    #for v in g.trainable_variables:
    #  print(v.name)
    #print('c.trainable_variables')
    #for v in c.trainable_variables:
    #  print(v.name)

    ### for tensorboard
    g1_loss(g_loss)
    d1_loss(new_d_loss)
    w_loss((-1)*(d_loss))  #wasserstein distance


    return 

def train(nsteps,ndims,lmbda,n_critic,batch_size,batches,training_data,input_to_GAN, epochs, g, c, g_opt, c_opt, g1_loss, d1_loss, w_loss, g1_summary_writer, d1_summary_writer, w_summary_writer):

    losses = np.zeros((epochs,4))

    for epoch in range(epochs):

        noise = input_to_GAN
        real_data = training_data #X1.astype('int')

        # uncommenting this line means that the noise is not paired with the outputs (probably desirable)
        #noise = np.random.normal(size=[noise.shape[0],noise.shape[1]])
 
        real_data, noise = sklearn.utils.shuffle(real_data, noise) #shuffle each epoch
        #print ('shuffled l_input1 and xn1_comp')
        
        xx1 = real_data.reshape(batches, batch_size, ndims*nsteps)
        inpt1 = noise.reshape(batches, batch_size, ndims)
        #print ('data arranged in batches')
        
        
        for i in range(len(xx1)):
            #print('calling train_step', i ,'of',len(xx1), '-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')  
            train_step(inpt1[i], xx1[i], lmbda, n_critic, g, c, g_opt, c_opt, g1_loss, d1_loss, w_loss)
            #print('back from train_step')  

        print('epoch:', epoch, '*************************************************')    
        print('gen loss', g1_loss.result().numpy(), 'd loss', d1_loss.result().numpy(), 'w_loss' , w_loss.result().numpy())

        losses[epoch,:] = [ epoch+1, g1_loss.result().numpy() ,   d1_loss.result().numpy(),  w_loss.result().numpy()]

        with g1_summary_writer.as_default():
            tf.summary.scalar('loss', g1_loss.result(), step=epoch)

        with d1_summary_writer.as_default():
            tf.summary.scalar('loss', d1_loss.result(), step=epoch)

        with w_summary_writer.as_default():
            tf.summary.scalar('loss', w_loss.result(), step=epoch)
            
        #print('reset states')
        g1_loss.reset_states()
        d1_loss.reset_states()
        w_loss.reset_states()

        if (epoch + 1) % 1000 == 0:
        #if epoch < 100 or (epoch + 1) % 100 == 0 :
                       
            saved_g1_dir = './saved_g_' + str(epoch + 1)
            saved_d1_dir = './saved_c_' + str(epoch + 1)
            tf.keras.models.save_model(g, saved_g1_dir)
            tf.keras.models.save_model(c, saved_d1_dir)

    np.savetxt('losses.csv', losses, delimiter=',')

    return g


def learn_hypersurface_from_POD_coeffs(input_to_GAN, training_data, nsteps, nPOD, ndims, lmbda, n_critic, batch_size, batches, ndims_latent_input):
    # nPOD not needed

    try:
      print('looking for previous saved models')
      saved_g1_dir = './saved_g_' + str(model_number)
      g = tf.keras.models.load_model(saved_g1_dir)

      saved_d1_dir = './saved_c_' + str(model_number)
      c = tf.keras.models.load_model(saved_d1_dir)


    except:
      print('making new generator and critic')
      g = make_generator(nsteps)
      c = make_critic(nsteps)


    g_opt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
    c_opt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)

    g1_loss = tf.keras.metrics.Mean('g1_loss', dtype=tf.float32)
    d1_loss = tf.keras.metrics.Mean('d1_loss', dtype=tf.float32)
    w_loss = tf.keras.metrics.Mean('w_loss', dtype=tf.float32)

    # logs to follow losses on tensorboard
    print('initialising logs for tensorboard')
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    g1_log_dir = './logs/gradient_tape/' + current_time + '/g'
    d1_log_dir = './logs/gradient_tape/' + current_time + '/d'
    w_log_dir = './logs/gradient_tape/' + current_time + '/w'

    g1_summary_writer = tf.summary.create_file_writer(g1_log_dir)
    d1_summary_writer = tf.summary.create_file_writer(d1_log_dir)
    w_summary_writer = tf.summary.create_file_writer(w_log_dir)


    print('beginning training')
    epochs = 500
    generator = train(nsteps,ndims,lmbda,n_critic,batch_size,batches,training_data,input_to_GAN, epochs, g, c, g_opt, c_opt, g1_loss, d1_loss, w_loss, g1_summary_writer, d1_summary_writer, w_summary_writer)
    print('ending training')


    # generate some random inputs and put through generator
    number_test_examples = 10
    test_input = tf.random.normal([number_test_examples, ndims_latent_input])
    predictions = generator(test_input, training=False)
    #predictions = generator.predict(test_input) # number_test_examples by ndims_latent_input

    predictions_np = predictions.numpy() # nExamples by nPOD*nsteps
    #tf.compat.v1.InteractiveSession()
    #predictions_np = predictions.numpy().
    print('Shape of the output of the GAN', predictions_np.shape)
    predictions_np = predictions_np.reshape(number_test_examples*nsteps, nPOD)
    print('Reshaping the GAN output (in order to apply inverse scaling)', predictions_np.shape)

    return predictions_np, generator


# -----------------------------------------------------------------------------------------------------------------------------
# reproducibility
np.random.seed(143)
tf.random.set_seed(143)

# read in data, reshape and normalise
lmbda = 10
n_critic = 5

batch_size = 20 # 32  
batches = 10    # 900 

ndims_latent_input = 5 # 128 # latent variables for GAN

# data settings
nsteps = 5  #number of consecutive timesteps in gan
ndims = 5 # == nPOD # 128 # reduced variables i.e. POD coefficients or AE latent variables

# reading in the data
print('Reading in the POD coeffs.')
#csv_data = np.loadtxt('/kaggle/input/fpc-204examples-5pod-without-ic/POD_coeffs_1_204.csv', delimiter=',')
csv_data = np.loadtxt('POD_coeffs_1_204.csv', delimiter=',')
csv_data = np.float32(csv_data)
print('type and shape (nPOD by nTrain) of POD coeffs from csv file', type(csv_data), csv_data.shape, csv_data.dtype)

nTrain = csv_data.shape[1]
nPOD = csv_data.shape[0]

csv_data = csv_data.T # nTrain by nPOD

# scaling the POD coeffs
scaling = sklearn.preprocessing.MinMaxScaler(feature_range=[-1,1])
print('shape csv data for the scaling', csv_data.shape) 
csv_data = scaling.fit_transform(csv_data)

# check that the columns are scaled between min and max values (-1,1)
for icol in range(csv_data.shape[1]):
    print('min and max of col, ', icol ,' of csv_data:', np.min(csv_data[:,icol]), np.max(csv_data[:,icol]) )

# create nsteps time levels for the training_data for the GAN
t_begin = 0
t_end = nTrain - nsteps + 1
training_data = np.zeros((t_end,nPOD*nsteps),dtype=np.float32) # nTrain by nsteps*nPOD # 'float32' or np.float32

for step in range(nsteps):
    #print ('training data - cols',step*nPOD,'to',(step+1)*nPOD )
    #print ('csv data - rows', t_begin+step ,'to', t_end+step )
    training_data[:,step*nPOD:(step+1)*nPOD] = csv_data[t_begin+step : t_end+step,:]

print('Shape of training data for the GAN', training_data.shape, training_data.dtype)

# GAN input
try:
    input_to_GAN = np.load('input_to_GAN.npy')
except:
    input_to_GAN = tf.random.normal([training_data.shape[0], ndims_latent_input])
    input_to_GAN = input_to_GAN.numpy()


t0 = time.time()
predictions, generator = learn_hypersurface_from_POD_coeffs(input_to_GAN, training_data, nsteps, nPOD, ndims, lmbda, n_critic, batch_size, batches, ndims_latent_input)
t_train = time.time() - t0

print('training time', t_train)

# rescale 
predictions = scaling.inverse_transform(predictions).T
print('shape of predictions before writing to file', predictions.shape)

np.savetxt('prediction_from_GAN.csv', predictions, delimiter=',')  #save gan input if continuing training after job ends

print('Time taken to learn the hypersurface: ', t_train)

f = open('log.txt',"a")
f.write('Time taken to train: %s \n' % str(t_train)  )
#    f.write( '%s ' % str(t_train) )
f.close()

# optimisation part -------------------------------------------------
print('optimisation part...')
# reproducibility
np.random.seed(98)
tf.random.set_seed(98)

mse = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(5e-3)
def mse_loss(inp, outp):
    return mse(inp, outp)

nLatent = ndims_latent_input
print('nsteps',nsteps,'nLatent',nLatent)

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

####################################################################

t0 = time.time()
print('training_data', training_data.shape)
start_from = 100
inn = training_data[start_from,:(nsteps-1)*nPOD].reshape(1, (nsteps - 1) * nLatent)
print('inn',inn.shape)
npredictions = 400
#nLatent = 5 #latent_input_size = 5
nepochs_optimiser = 5000
initial_comp = training_data[start_from,:(nsteps-1)*nPOD].reshape((nsteps - 1), nLatent)
flds = timesteps(initial_comp, inn, npredictions)
print('flds',flds.shape)

# rescale 
flds = scaling.inverse_transform(flds).T
print('shape of predictions before writing to file', flds.shape)

np.savetxt('optimised_prediction_from_GAN.csv', flds, delimiter=',')  #save gan input if continuing training after job ends

t_optimise = time.time() - t0

f = open('log.txt',"a")
f.write('Time taken to optimise: %s \n' % str(t_optimise)  )
f.close()

import sys
print('python',sys.version)
print('numpy', np.__version__)
print('tf', tf.__version__)

