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




# Define parameters
test_window_size = 32# number of dimentions to leave out at a time
n_tests = int(np.floor(np.size(full_data_matrix[train_indicies,:],1)/test_window_size)) #make window size to match data size: must give an int
plt.axis([0, np.size(full_data_matrix,1), 0, 1])
for i in range(n_tests):
    #crop out dimentions
    print('cropping')
    data_to_test = np.delete(full_data_matrix[train_indicies,:],np.add(np.arange(test_window_size),i*test_window_size),1)
    #DO PCA
    
    
    
    
    svm_object = svm.SVC(C=3.2, gamma=6e-06)
    print('scoring')
    scores = cross_val_score(svm_object, data_to_test, full_subClass_array[train_indicies], cv=3)
    score = np.mean(scores)
    print('plotting')
    plt.scatter(np.arange(i*test_window_size+test_window_size),[score]*test_window_size)   
plt.show()









