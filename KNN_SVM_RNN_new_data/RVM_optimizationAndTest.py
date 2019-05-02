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
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

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


#def rvc_param_selection(X, y, nfolds): #https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
#    gammas = [1e-8,1e-7,1e-6, 1e-5,1e-4,1e-3,1e-2]
#    param_grid = {'coef1' : gammas}
#    grid_search = GridSearchCV(RVC(kernel='rbf'), param_grid, cv=nfolds)
#    grid_search.fit(X, y)
#    grid_search.best_params_
#    return grid_search.best_params_
#
#def rvc_param_selection2(X, y, nfolds): #https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
#    gammas = [1e-4,1e-3,1e-2]
#    param_grid = {'coef1' : gammas}
#    grid_search = GridSearchCV(RVC(kernel='rbf'), param_grid, cv=nfolds)
#    grid_search.fit(X, y)
#    grid_search.best_params_
#    return grid_search.best_params_




#WE START with the biary case
    
#X = full_normPCA128_array[train_indicies]
#Y = full_isAnimal_array[train_indicies]
#
#params = rvc_param_selection(X,Y,5)
#
#params = rvc_param_selection2(X,Y,5)



####################TO TEST
clf = RVC(kernel="rbf",coef1=1e-5)# coef1:  1=46 0.1same 
clf.fit(full_normPCA128_array[train_indicies], full_isAnimal_array[train_indicies])
clf.score(full_normPCA128_array[train_indicies],full_isAnimal_array[train_indicies])
clf.score(full_normPCA128_array[test_indicies],full_isAnimal_array[test_indicies])


#predictions = clf.predict(full_normPCA128_array[test_indicies])
#
#corr=0
#for i in range(np.size(predictions)):
#    if predictions[i] == full_isAnimal_array[test_indicies][i]:
#        corr += 1
        