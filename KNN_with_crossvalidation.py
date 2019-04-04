# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:23:34 2019

@author: Ralle
"""

import scipy.io
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.decomposition import PCA
import os

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

def is_correct(pred,y_test):# returns percent of correct predictions
    n = np.size(y_test)
    percent_correct = 0
    for i in range(n):
        if (y_test[i] == 1) and (pred[i] > 0.5):
            percent_correct += 1
        elif (y_test[i] == 0) and (pred[i] <= 0.5):
            percent_correct += 1
    percent_correct = percent_correct/n
    return percent_correct

def PCA_func(input_data,dims): #for 2d data with one observation per row. not really neccesary to have as function
    pca = PCA(n_components=dims, svd_solver='full')
    pca.fit(input_data)
    return pca

os.chdir('C:/Users/Ralle/Desktop/Advanced Machine Learning Project/AML/Nicolai/data')#

nimg = 690
nsubjects = 15
nchannels = 32
nsamples = 550

##Task 1: load data and convert to 2D array
#create array for data
#create array for classes
full_data_matrix = []
full_class_array = []
for i in range(nsubjects):
    eeg_events = scipy.io.loadmat('exp' + str(i+1) + '\eeg_events.mat')
    image_order = np.genfromtxt('exp' + str(i+1) + '\image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#
    data = eeg_events["eeg_events"]
    concat_data = concat_channels(data)
    if i == 0:
        full_data_matrix = concat_data
        full_class_array = image_order[:,0]
    else:
        full_data_matrix  = np.concatenate((full_data_matrix,concat_data))
        full_class_array  = np.concatenate((full_class_array,image_order[:,0]))

##Task 2: Normalize
#Classes to 0 and 1
binary_classes_all = is_animal(full_class_array)
#Normalize data
normal_data_all = preprocessing.scale(full_data_matrix)#normalize

###################################Cut off some data
##downsample:
####???????

pca = PCA_func(normal_data_all, 100)#PCA with 100 components
normal_data = pca.transform(normal_data_all)#transform data to 100 components
binary_classes = binary_classes_all



##KNN 2 layer crossvalidation



n_models = 10
n_outer_splits = 10
n_inner_splits = 10
#K_neighbours from 1 to 10
kf = KFold(n_splits=n_outer_splits, shuffle = True)
kf.get_n_splits(normal_data)
nsamples = np.size(binary_classes)
test_err = 0
for train_index, test_index in kf.split(normal_data): #makes the outer split one subject left out
    print('outer loop started')
    E_gen_s = np.zeros((n_models,1))
    outer_trainX = normal_data[train_index,:]
    outer_testX = normal_data[test_index,:]
    outer_trainY = binary_classes[train_index]
    outer_testY = binary_classes[test_index]
    kf_inner = KFold(n_splits=n_inner_splits, shuffle = True)#Maybe shuffle on # one subject left out
    kf_inner.get_n_splits(outer_trainX)
    nDtest = np.size(outer_testY)
    for train_index_inner, test_index_inner in kf.split(outer_trainX): #inner split. one subject left out
        print('\t inner loop started')
        inner_trainX = outer_trainX[train_index_inner,:]
        inner_testX = outer_trainX[test_index_inner,:]
        inner_trainY = outer_trainY[train_index_inner]
        inner_testY = outer_trainY[test_index_inner]
        nDval = np.size(inner_testY)
        nDpar = np.size(outer_trainY)
        for i in range(n_models):#number of models 1 to 10 neighbours
            #print('\t \t training models loop started')
            n_neigh = i+1
            #train
            KNN_model = KNeighborsClassifier(n_neighbors=n_neigh)
            KNN_model.fit(inner_trainX,inner_trainY) 
            #validate
            pred = KNN_model.predict(inner_testX)
            err = 1-is_correct(pred,inner_testY)
            #print(err)
            E_gen_s[i] += err*nDval/nDpar
    n_opt = E_gen_s.argmin()+1
    print('optimal neighbours in outer loop')
    print(n_opt)
    KNN_model = KNeighborsClassifier(n_neighbors=n_opt)
    KNN_model.fit(outer_trainX,outer_trainY)
    pred = KNN_model.predict(outer_testX)
    test_err += (1-is_correct(pred,outer_testY))*nDtest/nsamples
    print('error in outerloop')
    print(1-is_correct(pred,outer_testY))
print('test error after all loops')
print(test_err)

