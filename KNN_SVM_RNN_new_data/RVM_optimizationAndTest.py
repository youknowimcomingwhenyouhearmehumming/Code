# -*- coding: utf-8 -*-
"""
Created on Wed May  1 10:12:17 2019

@author: Ralle
"""

import scipy.io
import numpy as np
import os
import pickle
from skrvm import RVC
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

#WE START with the biary case

####################TO TEST
clf = RVC(kernel='rbf',alpha=1e-08, beta=1e-08,)
clf.fit(full_normPCA128_array[train_indicies], full_isAnimal_array[train_indicies])
clf.score(full_normPCA128_array[test_indicies],full_isAnimal_array[test_indicies])

clf = RVC(kernel='rbf',alpha=1e-08, beta=1e-08,)
clf.fit(full_normPCA128_array[train_indicies], full_isAnimal_array[train_indicies])
clf.score(full_normPCA128_array[test_indicies],full_isAnimal_array[test_indicies])