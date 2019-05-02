# -*- coding: utf-8 -*-
"""
Created on Thu May  2 09:43:28 2019

@author: Ralle
"""
import scipy.io
import numpy as np
import os
import pickle
from skrvm import RVC
from sklearn.model_selection import GridSearchCV

###############RUN THE SCRIP KNN_DTW FIRST!!!!!

def is_correct(pred,y_test):# returns percent of correct predictions
    n = np.size(y_test)
    percent_correct = 0
    for i in range(n):
        if (y_test[i] == pred[i]):
            percent_correct += 1
    percent_correct = percent_correct/n
    return percent_correct


#from sklearn.model_selection import GridSearchCV

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
full_normPCA655_array = np.load('full_normPCA655_array.npy')
full_normPCA128_array = np.load('full_normPCA128_array.npy')
#######################################################################



####################KNN_DTW
knn_dtw_object = KnnDtw(max_warping_window=5)
knn_dtw_object.fit(full_data_matrix[train_indicies,0:576],full_isAnimal_array[train_indicies])


pred = knn_dtw_object.predict(full_data_matrix[test_indicies,0:576])

is_correct(pred[0],full_isAnimal_array[test_indicies])


