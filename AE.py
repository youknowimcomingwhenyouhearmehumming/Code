# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:30:20 2019
Variational autoencoder for Mindreaders project.
Two layer encoder and decoder for classification of EEG signals
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
from keras import backend as K
import os
import os.path
from Mindreader_functions import *


if not os.path.exists('\\weights'):
        os.mkdir('\\weights')
        print("Directory " , "\\weights" ,  " Created ")
else:    
        print("Directory " , "\\weights" ,  " already exists")
        
# Loading data
num_classes = 6
dataPreConcat, image_order = data_loader(4)
labels=one_hot_encoder(num_classes,dataPreConcat,image_order)
dataPostConcat = concat_channels(dataPreConcat)

print(dataPostConcat.shape)
print(labels.shape)

# Splitting test and training data and normalizing
x_train, x_test, y_train, y_test = train_test_split(dataPostConcat, np.transpose(labels), test_size=0.2, random_state=0)
x_train = x_train.astype('float32') / float(x_train.max())
x_test = x_test.astype('float32') / float(x_test.max())

print("x_train - min:", np.min(x_train), "max:",np.max(x_train))
print("x_test - min:", np.min(x_test), "max:",np.max(x_test))
# Model section start

encoding_dim = 32

compression_factor = float(x_train.shape[1]) / encoding_dim
print("Compression factor: %s" % compression_factor)

#input_vector = Input(shape = (x_train.shape[1],))
input_dim=(x_train.shape[1],)
input_vector = Input((36864,))


#Autoencoder model start

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
#    flat = Flatten()(encoder_output)
    den = Dense(64, activation='relu')(encoder_output)
    print("den:",den.shape)
    out = Dense(num_classes, activation='softmax')(den)
    print("out:",out.shape)
    return out


# Hyper parameters
epochs=2
batch_size=16


#autoencoder_train(epochs, batch_size, input_autoencoder):        
autoencoder = Model(input_vector, _decoder(_encoder(input_vector)))
#autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()
autoencoder_train = autoencoder.fit(x_train, x_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, x_test))
 
# Create target Directory if don't exist

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




#autoencoder_trained = autoencoder_train(epochs,batch_size,input_vector)

encode = _encoder(input_vector)
full_model = Model(input_vector,fc(encode))

for l1,l2 in zip(full_model.layers[0:7],autoencoder.layers[0:7]):
    l1.set_weights(l2.get_weights())

print(autoencoder.get_weights()[0][1])
print(full_model.get_weights()[0][1])
test1 = autoencoder.get_weights()[0][1]
test2 = full_model.get_weights()[0][1]
print(np.max(test1-test2))
#

full_model.save_weights('autoencoder_classification.h5')
for layer in full_model.layers[0:19]:
    layer.trainable = False
full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

full_model.summary()

classify_train = full_model.fit(x_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
full_model.save_weights('autoencoder_classification.h5')
#AutoEncoder model end
#autoencoder_train = autoencoder.fit(x_train, x_train,epochs=epochs,batch_size=batch_size,shuffle=True,validation_data=(x_test, x_test),verbose=1)


#decoded_vector = autoencoder.predict(x_test)


accuracy = classify_train.history['acc']
val_accuracy = classify_train.history['val_acc']
loss = classify_train.history['loss']
val_loss = classify_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


test_eval = full_model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
predicted_classes = full_model.predict(x_test)

predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
predicted_classes.shape, y_test.shape

#print( "Found %d correct labels" % len(correct))
#for i, correct in enumerate(correct[:9]):
#    plt.subplot(3,3,i+1)
#    plt.imshow(x_test[correct].reshape(28,28), cmap='gray', interpolation='none')
#    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
#    plt.tight_layout()
#    
#    
#    incorrect = np.where(predicted_classes!=y_test)[0]
#print( "Found %d incorrect labels" % len(incorrect))
#for i, incorrect in enumerate(incorrect[:9]):
#    plt.subplot(3,3,i+1)
#    plt.imshow(x_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
#    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))
#    plt.tight_layout()
#    
#    
#    
#from sklearn.metrics import classification_report
#target_names = ["Class {}".format(i) for i in range(num_classes)]
#print(classification_report(y_test, predicted_classes, target_names=target_names))