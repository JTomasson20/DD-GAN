"""
Some ownership comments
"""

import tensorflow as tf
import keras
import numpy as np


class GAN:
    """
    Generator discriminator etc
    """

    def __init__(self):
        """
        Stuff happens here
        """

        self.kwargs = None
        self.nsteps = None
        self.ndims = None
        self.lmbda = None
        self.n_critic = None
        self.batch_size = None # 32  
        self.batches = None    # 900 

        self.generator = tf.keras.Sequential()
        self.discriminator = tf.keras.Sequential()


    def setup(self, kwargs) -> None:
        self.kwargs = kwargs
        self.nsteps = kwargs.pop("nsteps", 5)
        self.ndims = kwargs.pop("ndims", 5)
        self.lmbda = kwargs.pop("lambda", 10)
        self.n_critic = kwargs.pop("n_critic", 5)
        self.batch_size = kwargs.pop("batch_size", 20) # 32  
        self.batches = kwargs.pop("batches", 10) # 900

        self.make_generator()
        self.make_discriminator()


    def make_generator(self): #nsteps):
        """
        Generator
        """
        self.generator.add(tf.keras.layers.Dense(5, input_shape=(5,), activation='relu')) # 5
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.Dense(10, activation='relu'))                  # 10
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.Dense((5*self.nsteps), activation='relu'))     # 25
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.Dense((5*nsteps), activation='tanh'))          # 25
        return None


    def make_discriminator(self): #nsteps):
        """
        Discriminator
        """

        self.discriminator.add(tf.keras.layers.Dense(5*self.nsteps, input_shape=(5*nsteps,)))
        self.discriminator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.discriminator.add(tf.keras.layers.Dropout(0.3))
        self.discriminator.add(tf.keras.layers.Dense(10))
        self.discriminator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.discriminator.add(tf.keras.layers.Dense(5))
        self.discriminator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.discriminator.add(tf.keras.layers.Dropout(0.3))
        self.discriminator.add(tf.keras.layers.Flatten())
        self.discriminator.add(tf.keras.layers.Dense(1))
        return None


    def discriminator_loss(self, d_real, d_fake):
        d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
        return d_loss

    def generator_loss(self, d_fake):
        g_loss = -tf.reduce_mean(d_fake)
        return g_loss


@tf.function
def train_step(noise, real, lmbda, n_critic, g, c, g_opt, c_opt, g1_loss, d1_loss, w_loss):   
    batch_size = len(real)

    for i in range(n_critic):    
        with tf.GradientTape() as t:
            with tf.GradientTape() as t1:
                fake = g(noise, training=True) # training=False?
                d_real = c(real, training=True)
                d_fake = c(fake, training=True)
                d_loss = discriminator_loss(d_real, d_fake)   #initial c loss


                epsilon = tf.random.uniform(shape=[batch_size, 1], minval=0., maxval=1.)
                #print ('types', real.dtype, epsilon.dtype, fake.dtype)                
                interpolated = real + epsilon * (fake - real)  
                t1.watch(interpolated)
                c_inter = c(interpolated, training=True)  


                
            grad_interpolated = t1.gradient(c_inter, interpolated)
            slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_interpolated) + 1e-12, axis=[1])) # 
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            new_d_loss = d_loss + (lmbda*gradient_penalty)  #new c loss
        
        c_grad = t.gradient(new_d_loss, c.trainable_variables)
        c_opt.apply_gradients(zip(c_grad, c.trainable_variables))


    ##### train generator #####

    with tf.GradientTape() as gen_tape:
        fake_images = g(noise, training=True)
        d_fake = c(fake_images, training=True) # training=False?
        g_loss = generator_loss(d_fake)

    gen_grads = gen_tape.gradient(g_loss, g.trainable_variables)
    g_opt.apply_gradients(zip(gen_grads, g.trainable_variables))


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
        
        xx1 = real_data.reshape(batches, batch_size, ndims*nsteps)
        inpt1 = noise.reshape(batches, batch_size, ndims)
        
        
        for i in range(len(xx1)):
            train_step(inpt1[i], xx1[i], lmbda, n_critic, g, c, g_opt, c_opt, g1_loss, d1_loss, w_loss)

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
    