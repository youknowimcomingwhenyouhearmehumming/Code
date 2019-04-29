# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:59:57 2019

@author: Ralle
"""

import scipy.io
import numpy as np
import os
import matplotlib.pyplot as plt
from skrvm import RVC
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split



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

def class2numbers(class_array):
    classes = np.unique(class_array)
    new_class_array = np.zeros((np.size(class_array),1))
    for i in range(np.size(classes,0)):
        for j in range(np.size(class_array,0)):
            if class_array[j] == classes[i]:
                new_class_array[j] = i
    new_class_array = np.ravel(new_class_array)
    return new_class_array, classes


os.chdir('C:/Users/Ralle/Desktop/Advanced Machine Learning Project/AML/Nicolai/data')#
nsubjects = 15

full_data_matrix = []
full_class_array = []
full_semantics_matrix = []
for i in range(nsubjects):
    eeg_events = scipy.io.loadmat('exp' + str(i+1) + '\eeg_events.mat')
    image_order = np.genfromtxt('exp' + str(i+1) + '\image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#
    data = eeg_events["eeg_events"]
    concat_data = concat_channels(data)
    
    #load sematics
    image_semantics_mat = scipy.io.loadmat('exp' + str(i+1) + '\image_semantics.mat')
    image_semantics = image_semantics_mat["image_semantics"]#semantics_vector x n_images
    
    
    
    if i == 0:
        full_data_matrix = concat_data
        full_class_array = image_order[:,1]
        full_semantics_matrix = np.transpose(image_semantics)
    else:
        full_data_matrix  = np.concatenate((full_data_matrix,concat_data))
        full_class_array  = np.concatenate((full_class_array,image_order[:,1]))
        full_semantics_matrix = np.concatenate((full_semantics_matrix,np.transpose(image_semantics)))
        

#only p1
p1_data = full_data_matrix[0:690]
p1_classes = full_class_array[0:690]



#normalize
p1_normal_data = preprocessing.scale(p1_data)#normalize

#PCA
pca = PCA(10, svd_solver='auto')
pca.fit(p1_normal_data)
p1_pca_data = pca.transform(p1_normal_data)#transform data to xx components

p1_pca_data
#classes_numbered, class_numbers_names = class2numbers(p1_classes)

## RVM classification
clf1=RVC()
clf1.fit(p1_pca_data,p1_classes)

pred = clf1.predict(p1_pca_data)

correct = 0
for i in range(np.size(pred,0)):
    if pred[i] == p1_classes[i]:
        correct += 1

clf1

params = clf1.get_params
params.alpha_

clf1.predict_proba


















