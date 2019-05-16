# -*- coding: utf-8 -*-
"""
Created on Sun May 12 15:57:22 2019

@author: JAlbe
"""
import numpy as np
import scipy.io as sio
from tensorflow import set_random_seed
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, Input, MaxPooling1D, Conv1D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.losses import mse, binary_crossentropy, categorical_crossentropy
from keras.utils import plot_model


def _encoder(encoder_input):
    # Encoder Layers
    print(encoder_input.shape)
    print(input_dim)
#    layer1 = Dense(4 * encoding_dim, input_shape=(input_dim,), activation='relu')(encoder_input)
    layer1 = Dense(units=4 * encoding_dim, activation='relu')(encoder_input)
    layer1 = BatchNormalization()(layer1)
    print("layer1:",layer1.shape)
    layer2 = Dense(2 * encoding_dim, activation='relu')(layer1)
    layer2 = BatchNormalization()(layer2)
    print("layer2:",layer2.shape)
    encoder_output = Dense(2 * encoding_dim, activation='relu')(layer2)
    encoder_output = BatchNormalization()(layer2)
    print("encoder_output:",encoder_output.shape)
    
    return encoder_output

def _decoder(encoder_output):
    # Decoder Layers
    layer1 = Dense(2 * encoding_dim, activation='relu')(encoder_output)
    layer1 = BatchNormalization()(layer1)
    print("layer1:",layer1.shape)
    layer2 = Dense(4 * encoding_dim, activation='relu')(layer1)
    layer2 = BatchNormalization()(layer2)    
    print("layer2:",layer2.shape)
    decoder_output = Dense(36864, activation='sigmoid')(layer2)
    print(decoder_output.shape)
    print("decoder_output:",decoder_output.shape)
    return decoder_output

def fc(encoder_output):
    flat = Flatten()(encoder_output)
    den = Dense(128, activation='relu')(flat)
    out = Dense(num_classes, activation='softmax')(den)
    return out

def autoencoder_train(epochs, batch_size, input_autoencoder):        
    autoencoder = Model(input_autoencoder, _decoder(_encoder(input_autoencoder)))
    #autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')
    autoencoder.summary()
    autoencoder_train = autoencoder.fit(x_train, x_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, x_test))
    autoencoder.save_weights('autoencoder.h5')
    
    loss = autoencoder_train.history['loss']
    val_loss = autoencoder_train.history['val_loss']   
    epochs = range(epochs)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
