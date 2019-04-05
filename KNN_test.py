# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:26:33 2019

@author: Ralle
"""

import scipy.io
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing

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

def is_animal(class_vector):#creates vector of 1 if animal and 0 if not
    n = np.size(class_vector)
    bin_vector = np.zeros(n)
    for i in range(n):
        if class_vector[i] == 'animal':
            bin_vector[i] = 1
        else:
            bin_vector[i] = 0
    return bin_vector


def is_correct(pred,y_test):# returns number of correct predictions
    n = np.size(y_test)
    percent_correct = 0
    for i in range(n):
        if (y_test[i] == 1) and (pred[i] > 0.5):
            percent_correct += 1
        elif (y_test[i] == 0) and (pred[i] <= 0.5):
            percent_correct += 1
    percent_correct = percent_correct/n*100
    return percent_correct

def PCA_func(input_data,dims): #for 2d data with one observation per row. not really neccesary to have as function
    pca = PCA(n_components=dims, svd_solver='full')
    pca.fit(input_data)
    return pca

def do_KNN(X_train, X_test, y_train, y_test, n_neigh):
    neigh = KNeighborsClassifier(n_neighbors=n_neigh)
    neigh.fit(X_train,y_train)
    pred = neigh.predict(X_test)
    return is_correct(pred,y_test)


eeg_events = scipy.io.loadmat('AML/Nicolai/data/exp1/eeg_events.mat')
image_order = np.genfromtxt('AML/Nicolai/data/exp1/image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#

nChannels,nSamples,nImg = np.shape(eeg_events["eeg_events"])
    
data = eeg_events["eeg_events"]
new_data = concat_channels(data)#merge channels
new_data_normal = preprocessing.scale(new_data)#normalize

pca = PCA_func(new_data_normal, 100)#PCA with 100 components
data_transformed = pca.transform(new_data_normal)#transform data to 100 components

var_explained = pca.explained_variance_ratio_
plt.figure()
plt.plot(var_explained)
plt.show()

##########try 100 different sets(i think they are different)
#and mean the correct prediction
#k neighbours is set to 5
sum = 0
k_neighbours = 5
for i in range(100): 
    X_train_index, X_test_index, y_train_string, y_test_string = train_test_split(list(range(nImg)),image_order[:,0],test_size=0.2)
    X_train = data_transformed[X_train_index,:]
    X_test = data_transformed[X_test_index,:]
    y_train = is_animal(y_train_string)
    y_test = is_animal(y_test_string)
    sum = (do_KNN(X_train, X_test, y_train, y_test, k_neighbours))
print(sum/100)



