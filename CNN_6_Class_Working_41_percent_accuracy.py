# -*- coding: utf-8 -*-
"""
Created on Tue May 14 22:18:36 2019

@author: JAlbe
"""
from __future__ import print_function

#import mne
import numpy as np
import scipy.io as sio
import scipy
import sklearn as sk
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import MaxPooling1D, Conv1D, Flatten
from keras.layers.normalization import BatchNormalization
from Mindreader_functions import *
from sklearn import preprocessing
import os
from sklearn.metrics import classification_report,confusion_matrix,precision_score
from sklearn.decomposition import PCA
import pickle


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


#os.chdir(r"C:\Users\JAlbe\Desktop\Mindreaderfolder\data")
os.chdir(r"C:\Users\JAlbe\Desktop\Mindreaders\Mindreaderfolder")

n_experiments = 4


full_data_matrix = []
full_superClass_array = []
full_subClass_array = []
full_semantics_matrix = []

for i in range(n_experiments):
    sub = 'data\exp'+str(i+1)+'\eeg_events.mat'
    sem = 'data\exp'+str(i+1)+'\image_semantics.mat'
    img_order = 'data\exp'+str(i+1)+'\image_order.txt'
    
    
    eeg_events = scipy.io.loadmat(sub)
    image_order = np.genfromtxt(img_order, delimiter="\t", skip_header = True, dtype=(str))#
    data = eeg_events["eeg_events"]
    concat_data = concat_channels(data)
    #load sematics
    image_semantics_mat = scipy.io.loadmat(sem)
    image_semantics = image_semantics_mat["image_semantics"]#semantics_vector x n_images
    
    if i == 0:
        full_data_matrix = concat_data
        full_superClass_array = image_order[:,0]
        full_subClass_array = image_order[:,1]
        full_semantics_matrix = np.transpose(image_semantics)

    else:
        full_data_matrix  = np.concatenate((full_data_matrix,concat_data),axis=0)
        full_superClass_array  = np.concatenate((full_superClass_array,image_order[:,0]),axis=0)
        full_subClass_array  = np.concatenate((full_subClass_array,image_order[:,1]),axis=0)
        full_semantics_matrix = np.concatenate((full_semantics_matrix,np.transpose(image_semantics)))
        
#Normalize data
  
normal_data_all = preprocessing.scale(full_data_matrix)#normalize

pca = PCA(2160, svd_solver='auto')
pca.fit(normal_data_all)
normal_data_pca = pca.transform(normal_data_all)

batch_size = 256
epochs = 100
num_classes = 6


n_observations =2160
x_train, x_test, y_train_index, y_test_index = train_test_split(normal_data_all[range(n_observations),:],range(n_observations),test_size=0.2) #The reason why y-labels are not sorte least to largest is because this funktion mix things arround in order to get a mo

y_train_string=full_subClass_array[y_train_index]

y_test_string=full_subClass_array[y_test_index]


"""
6 label
"""
y_train=np.zeros(len(y_train_string))
y_test=np.ceil(np.zeros(len(y_test_string)))

for i in range(len(y_train_string)):
    if y_train_string[i]=='airplane':
        y_train[i]=1
    if y_train_string[i]=='elephant':
        y_train[i]=2
    if y_train_string[i]=='pizza':
        y_train[i]=3
    if y_train_string[i]=='sheep':
        y_train[i]=4
    if y_train_string[i]=='train':
        y_train[i]=5
    if y_train_string[i]=='zebra':
        y_train[i]=0

for i in range(len(y_test_string)):
    if y_test_string[i]=='airplane':
        y_test[i]=1
    if y_test_string[i]=='elephant':
        y_test[i]=2
    if y_test_string[i]=='pizza':
        y_test[i]=3
    if y_test_string[i]=='sheep':
        y_test[i]=4
    if y_test_string[i]=='train':
        y_test[i]=5
    if y_test_string[i]=='zebra':
        y_test[i]=0

#for i in range(len(y_train_string)):
#    if y_train_string[i]=='airplane':
#        y_train[i]=0
#    if y_train_string[i]=='elephant':
#        y_train[i]=0
#    if y_train_string[i]=='pizza':
#        y_train[i]=1
#    if y_train_string[i]=='sheep':
#        y_train[i]=0
#    if y_train_string[i]=='train':
#        y_train[i]=0
#    if y_train_string[i]=='zebra':
#        y_train[i]=0
#
#for i in range(len(y_test_string)):
#    if y_test_string[i]=='airplane':
#        y_test[i]=0
#    if y_test_string[i]=='elephant':
#        y_test[i]=0
#    if y_test_string[i]=='pizza':
#        y_test[i]=1
#    if y_test_string[i]=='sheep':
#        y_test[i]=0
#    if y_test_string[i]=='train':
#        y_test[i]=0
#    if y_test_string[i]=='zebra':
#        y_test[i]=0


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
x_test = np.expand_dims(x_test, 2)
## convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
#x_train = x_train.reshape((483,17600,-1))

model = Sequential()
model.add(Conv1D(8, 3,strides=1, padding='valid',input_shape=(x_train[0].shape[0],1),data_format="channels_last",activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(4, 3, strides=1, padding='valid',activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=1e-1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-15, amsgrad=False),
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.save('CNNWorkingModel.h5')
model.save_weights('CNNWorkingModelWeights.h5')

import math
#error_rate= (float(score[1]))
error_rate = 0.73
#num_obs = (float(x_test.size[0]))
num_obs = 2160
uncert = math.sqrt((error_rate*(1-error_rate))/(num_obs))

