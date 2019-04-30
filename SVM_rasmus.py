# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 09:33:14 2019

@author: Ralle
"""
from sklearn import svm
from sklearn.model_selection import cross_val_score
import scipy.io
import numpy as np
import os
import matplotlib.pyplot as plt
from skrvm import RVR
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

def is_animal(class_vector):#creates vector of 1 if animal and 0 if not
    n = np.size(class_vector)
    bin_vector = np.zeros(n)
    for i in range(n):
        if class_vector[i] == 'animal':
            bin_vector[i] = 1
        else:
            bin_vector[i] = 0
    return bin_vector




os.chdir('C:/Users/Ralle/Desktop/Advanced Machine Learning Project/AML/Nicolai/data')#
nsubjects = 15

#Load all data and classes
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
        full_class_array = image_order[:,1]#0 for superclass 1 for subclass
        full_semantics_matrix = np.transpose(image_semantics)
    else:
        full_data_matrix  = np.concatenate((full_data_matrix,concat_data))
        full_class_array  = np.concatenate((full_class_array,image_order[:,1]))#0 for superclass 1 for subclass
        full_semantics_matrix = np.concatenate((full_semantics_matrix,np.transpose(image_semantics)))
        
#only use 2 first subjects
p12_data = full_data_matrix[0:690*15]
p12_classes = full_class_array[0:690*15]

#Normalize
p12_normal_data = preprocessing.scale(p12_data)#normalize

# TRAIN TEST SPLIT
X_train, X_test, y_train_index, y_test_index = train_test_split(p12_normal_data,range(690*15),test_size=0.25)

#PCA
#pca = PCA(5, svd_solver='auto')
#pca.fit(X_train)
#p12_pca_data_train = pca.transform(X_train)#transform data to xx components
#p12_pca_data_test = pca.transform(X_test)#transform data to xx components

pca = PCA(100, svd_solver='auto')
pca.fit(p12_normal_data)
p12_pca_data = pca.transform(p12_normal_data)#transform data to xx components


X_test = pca.transform(X_test)#transform data to xx components
X_train = pca.transform(X_train)#transform data to xx components





#Animal or not animal
p12_class_animal = is_animal(p12_classes)


clf = svm.SVC(kernel='rbf', C=1)
scores = cross_val_score(clf, p12_pca_data, p12_class_animal, cv=5)



#C=80  gamma = 1e-5 #100PCA subclasses
#C=1.15 gamma=8e-5 #100pca animal/not


X_train, X_test, y_train_index, y_test_index = train_test_split(p12_normal_data,range(690*15),test_size=0.25)

aaa = svm.SVC(gamma=1e-5, C=80)

scores = cross_val_score(aaa, p12_pca_data, p12_classes, cv=5)

np.mean(scores)

np.shape(np.unique(p12_classes))
