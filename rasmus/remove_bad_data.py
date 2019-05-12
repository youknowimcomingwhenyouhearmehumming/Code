# -*- coding: utf-8 -*-
"""
Created on Fri May 10 09:44:25 2019

@author: Ralle
"""

import scipy.io
import numpy as np
import os
import pickle
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn import preprocessing

#For the test of comparing KNN/SVM/RVM basic


#First we get load the splits indicies and data
#To load:
os.chdir('C:/Users/Ralle/OneDrive/Dokumenter/GitHub/Code/KNN_SVM_RNN_new_data')

with open("train_indexes.txt", "rb") as fp:   # Unpickling
    train_indicies = pickle.load(fp)
with open("test_indexes.txt", "rb") as fp:   # Unpickling
    test_indicies = pickle.load(fp)

full_isAnimal_array = np.load('full_isAnimal_array.npy')
full_subClass_array = np.load('full_subClass_array.npy')
#full_normPCA600_array = np.load('full_normPCA600.npy')
#full_normPCA123_array = np.load('full_normPCA123.npy')


# full_data_matrix[train_indicies,:]



full_normalized_array = preprocessing.scale(full_data_matrix)#normalize


# Define parameters
test_window_size = 32# number of dimentions to leave out at a time
n_tests = int(np.floor(np.size(full_normalized_array[train_indicies,:],1)/test_window_size)) #make window size to match data size: must give an int
plt.axis([0, np.size(full_data_matrix,1), 0, 1])
all_scores = np.zeros((np.size(full_data_matrix,1)))
for i in range(n_tests):
    #crop out dimentions
    print('cropping')
    data_to_test = np.delete(full_normalized_array[train_indicies,:],np.add(np.arange(test_window_size),i*test_window_size),1)
    #DO PCA
    print('PCA')
    pca = PCA(svd_solver='auto', n_components = 123)#PCA with all components
    pca.fit(data_to_test)
    train_normPCA_array = pca.transform(data_to_test)
    
    svm_object = svm.SVC(C=3.2, gamma=6e-06)
    print('scoring')
    scores = cross_val_score(svm_object, train_normPCA_array, full_subClass_array[train_indicies], cv=3)
    score = np.mean(scores)
    all_scores[i*test_window_size:i*test_window_size+test_window_size] = [score]*test_window_size
    print('plotting')
    plt.scatter(np.add(np.arange(test_window_size),test_window_size*i),[score]*test_window_size)  
    plt.pause(0.05)
plt.show()








os.chdir('C:/Users/Ralle/OneDrive/Dokumenter/GitHub/Code/rasmus')
full_isAnimal_array = np.load('all_scores_32size_bins.npy')

#####################################################
##Remove one bin at a time and test
plt.figure()
dims_to_del = np.zeros((n_tests,test_window_size))
all_del_scores = np.zeros((n_tests))
for i in range(n_tests):
    #The highest scores is the least nessesary bins
    dims_to_del[i,:] = np.where(all_scores == np.amax(all_scores))[0]
    data_to_test = np.delete(full_normalized_array[train_indicies,:],dims_to_del[i,:],1)
    
#DO PCA
    print('PCA')
    pca = PCA(svd_solver='auto', n_components = 123)#PCA with all components
    pca.fit(data_to_test)
    train_normPCA_array = pca.transform(data_to_test)
    
    svm_object = svm.SVC(C=3.2, gamma=6e-06)
    print('scoring')
    scores = cross_val_score(svm_object, train_normPCA_array, full_subClass_array[train_indicies], cv=3)
    score = np.mean(scores)
    all_del_scores[i*test_window_size:i*test_window_size+test_window_size] = [score]*test_window_size
    print('plotting')
    plt.scatter(i,score)  
    plt.pause(0.05)
plt.show()
# Get the indices of maximum element in numpy array





















