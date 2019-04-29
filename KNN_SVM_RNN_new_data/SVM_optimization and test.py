# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:19:30 2019

@author: Ralle
"""
import scipy.io
import numpy as np
import os
import pickle


#For the test of comparing KNN/SVM/RVM basic


#First we get load the splits indicies and data
#To load:
os.chdir('C:/Users/Ralle/Documents/GitHub/AdvancedMachineLearning/KNN_SVM_RNN_new_data')

with open("train_indexes.txt", "rb") as fp:   # Unpickling
    train_indicies = pickle.load(fp)
with open("test_indexes.txt", "rb") as fp:   # Unpickling
    test_indicies = pickle.load(fp)

full_isAnimal_array = np.load('full_isAnimal_array.npy')
full_normPCA_array = np.load('full_normPCA_array.npy')
full_subClass_array = np.load('full_subClass_array.npy')

#


#Baseline with random guessing


#For KNN:
#Do cross validation to find of classification error


#FOR SVM
#Optimize parameters on the training set
#Get error from test set


#FOR RVM
#Optimize parameters on the training set
#Get error from test set