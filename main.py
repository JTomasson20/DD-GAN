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
