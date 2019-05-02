from __future__ import print_function

#import mne
import numpy as np
import scipy.io as sio
import sklearn as sk
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import MaxPooling1D, Conv1D, Flatten
from keras.layers.normalization import BatchNormalization

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
    if img[0] == 'animal':
        labels[i]= 1
    elif img[0]=='vehicle':
        labels[i]= 2
    else:
        labels[i]= 0
    i=i+1
        
print(labels[1:10], image_order[1:10,0])
    
batch_size = 10
num_classes = 3
epochs = 200

print(data.shape, labels.shape)

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
x_test = np.expand_dims(x_test, 2)
## convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
#x_train = x_train.reshape((483,17600,-1))

model = Sequential()
#model.add(Dense(32,input_shape=(36864,), activation='relu'))
model.add(Conv1D(16, 2,strides=4, padding='valid',input_shape=(36864,1),data_format="channels_last",activation='relu'))
model.add(MaxPooling1D(pool_size=2))
#model.add(keras.layers.ZeroPadding1D(2, input_shape=x_train.shape[:]))
#model.add(Conv1D(32, 2, strides=1, padding='valid',input_shape=(17600,1)))
model.add(Conv1D(4, 2, strides=2, padding='valid',activation='relu'))
model.add(MaxPooling1D(pool_size=2))
#model.add(Conv1D(32, 2, strides=1, padding='valid'))
model.add(Flatten())
model.add(BatchNormalization())
#model.add(Dense(512, activation='relu',input_shape=x_train.shape[1:]))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))


model.summary()



model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False),
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