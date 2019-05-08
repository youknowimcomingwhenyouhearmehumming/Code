# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:41:01 2019

@author: Ralle
"""

#SVM AND TSFRESH FOR MY BOYZ


import scipy.io
import numpy as np
import os
import pickle
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA


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
full_normPCA600_array = np.load('full_normPCA600.npy')
full_normPCA123_array = np.load('full_normPCA123.npy')


os.chdir('C:/Users/Ralle/OneDrive/Dokumenter/GitHub/Code/tsfresh feature extraction from timeseries')
extracted_features = np.load('tsfresh_exstracted_from_concatChanels.npy')


def is_correct(pred,y_test):# returns percent of correct predictions
    n = np.size(y_test)
    percent_correct = 0
    for i in range(n):
        if (y_test[i] == pred[i]):
            percent_correct += 1
    percent_correct = percent_correct/n
    return percent_correct





def svc_param_selection(X, y, nfolds): #https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
    Cs = [1e-2,1e-1,1e-0,1e+1,1e+2,1e+3]
    gammas = [1e-8,1e-7,1e-6, 1e-5,1e-4,1e-3,1e-2]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


def svc_param_selection2(X, y, nfolds, Cs, gammas): #https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


print('first')


full_normalized_array = preprocessing.scale(extracted_features)#normalize

############################ PCA_ 290 components holds 95% of variance
pca = PCA(svd_solver='auto', n_components = 290)#PCA with all components
pca.fit(full_normalized_array[train_indicies])
full_normPCA_array = pca.transform(full_normalized_array)



#FIRST
params1 = svc_param_selection(full_normPCA_array[train_indicies],full_isAnimal_array[train_indicies] , 5)
C1 = params1['C']
gamma1 = params1['gamma']#{'C': 100.0, 'gamma': 1e-05}




Cs = [1e-0,5,1e+1,2e+1,3e+1,4e+1,5e+1,6e+1,7e+1,8e+1,9e+1,1e+2,2e+2,3e+2,4e+2,5e+2]
gammas = [1e-6, 3e-6, 5e-6, 7e-6, 9e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5]

params2 = svc_param_selection2(full_normPCA_array[train_indicies],full_isAnimal_array[train_indicies] , 5,Cs,gammas)
#{'C': 200.0, 'gamma': 7e-06}


svm_object = svm.SVC(C=200, gamma=7e-06,probability=True)
svm_object.fit(full_normPCA_array[train_indicies],full_isAnimal_array[train_indicies])

svm_object.score(full_normPCA_array[test_indicies],full_isAnimal_array[test_indicies])
predictions = svm_object.predict(full_normPCA_array[test_indicies])
corr = is_correct(predictions,full_isAnimal_array[test_indicies])
N = np.size(full_isAnimal_array[test_indicies])
accuracy = np.sqrt(corr*(100-corr)/(100*N))























###################################################################################################

###############FOR ALL CLASSES



#FIRST
params1 = svc_param_selection(full_normPCA_array[train_indicies],full_subClass_array[train_indicies] , 5)
C1 = params1['C']
gamma1 = params1['gamma']#{'C': 10.0, 'gamma': 0.0001}


Cs = [1e-1,5,1e+0,2e+0,3e+0,4e+0,5e+0,6e+0,7e+0,8e+0,9e+0,1e+1,2e+1,3e+1,4e+1,5e+1]
gammas = [1e-5, 3e-5, 5e-5, 7e-5, 9e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4]

params2 = svc_param_selection2(full_normPCA_array[train_indicies],full_subClass_array[train_indicies] , 5,Cs,gammas)
#{'C': 40.0, 'gamma': 1e-05}






svm_object = svm.SVC(C=40, gamma=1e-5,probability=True)
svm_object.fit(full_normPCA_array[train_indicies],full_subClass_array[train_indicies])

svm_object.score(full_normPCA_array[test_indicies],full_subClass_array[test_indicies])
predictions = svm_object.predict(full_normPCA_array[test_indicies])
corr = is_correct(predictions,full_subClass_array[test_indicies])
N = np.size(full_subClass_array[test_indicies])
accuracy = np.sqrt(corr*(100-corr)/(100*N))




















