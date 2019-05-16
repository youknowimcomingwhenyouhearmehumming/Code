from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Input
from keras.layers import MaxPooling1D, Conv1D, Flatten
from keras.layers.normalization import BatchNormalization

from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K


import scipy.io as sio
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import argparse

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


def concat_channels(eeg_events):#channels*EEG_value*img 
    #for putting all channels in a single vector. return is n_img x 17600
    n_channels,n_samples,n_img = np.shape(eeg_events)
    concat_all = np.zeros((n_img,n_channels*n_samples))
    for i in range(n_img): #for each image
        concat_row = []
        for j in range(n_channels): #for each channel of channels
            #concat_data[i] = np.concatenate((concat_data[i],eeg_events[j,:,i]),axis=1)
            concat_row = np.concatenate((concat_row,eeg_events[j,:,i]))
        concat_all[i] = concat_row
    return concat_all #[n_img*17600]



mat_data_subject1=sio.loadmat(r"data\exp1\eeg_events.mat")
mat_data_semantics1=sio.loadmat(r"data\exp1\image_semantics.mat")
mat_data_imorder1=sio.loadmat(r"data\exp1\image_order.mat")
image_order = np.genfromtxt(r'data\exp1\image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#

#import the matrix of eeg events 64x1.125x512 of one specific subject
#there are 540 labels 6x 
data_1=mat_data_subject1['eeg_events']
data = concat_channels(data_1)
print(data.shape)
labels=np.zeros((540,), dtype=int)
i=0
for img in image_order:
    if img[1] == 'pizza':
        labels[i]= 1
    elif img[1]=='train':
        labels[i]= 2
    elif img[1]=='aeroplane':
        labels[i]= 3
    elif img[1]=='elephant':
        labels[i]= 4   
    elif img[1]=='sheep':
        labels[i]= 5
    else:
        labels[i]= 0
    i=i+1
        

    
batch_size = 10
num_classes = 6
epochs = 200



x_train, x_test, y_train, y_test = train_test_split(data, np.transpose(labels), test_size=0.2, random_state=0)
#x_train, x_test, y_train, y_test = train_test_split(data, image_order[:,0], test_size=0.3, random_state=0)
    

# the data, shuffled and split between train and test sets


# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')

#
x_train = np.expand_dims(x_train, 2)
print(x_train.shape, 'train samples')
x_test = np.expand_dims(x_test, 2)
## convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

##model.add(Conv1D(16, 2,strides=4, padding='valid',input_shape=(36864,1),data_format="channels_last",activation='relu'))
##---------------------------------------------------------------------
## network parameters
##input_shape = (original_dim, )
#intermediate_dim = 512
#batch_size = 128
#latent_dim = 6
#epochs = 50
#
## VAE model = encoder + decoder
## build encoder model
##
##inputs = Input(shape=input_shape, name='encoder_input')
#inputs = Input(shape=(36864,1), name='encoder_input')
#x = Dense(2*intermediate_dim, activation='relu')(inputs)
#
#z_mean = Dense(latent_dim, name='z_mean')(x)
#z_log_var = Dense(latent_dim, name='z_log_var')(x)
#print(z_mean,'\n', z_log_var)
#
## use reparameterization trick to push the sampling out as input
## note that "output_shape" isn't necessary with the TensorFlow backend
#z = Lambda(sampling, output_shape=(latent_dim,), name='z' )([z_mean, z_log_var])
#
## instantiate encoder model
#encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
#encoder.summary()
#plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
#
## build decoder model
#latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
#x = Dense(intermediate_dim, activation='relu')(latent_inputs)
#outputs = Dense((36864,1), activation='sigmoid')(x)
#
## instantiate decoder model
#decoder = Model(latent_inputs, outputs, name='decoder' )
#decoder.summary()
#plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
#
## instantiate VAE model
#outputs = decoder(encoder(inputs)[2])
#vae = Model(inputs, outputs, name='vae_mlp')
##-------------------------------------------------------------------------------------------------------------------
