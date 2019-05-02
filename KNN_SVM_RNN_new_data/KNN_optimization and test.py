# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:21:02 2019

@author: Ralle
"""

##KNN MOFOS
import scipy.io
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.decomposition import PCA
import os
from sklearn.model_selection import GridSearchCV
import scipy.io
import numpy as np
import os
import pickle
from sklearn import svm



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


def is_correct(pred,y_test):# returns percent of correct predictions
    n = np.size(y_test)
    percent_correct = 0
    for i in range(n):
        if (y_test[i] == pred[i]):
            percent_correct += 1
    percent_correct = percent_correct/n
    return percent_correct


def knn_param_selection(X, y, nfolds): #https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
    k = [14,15,16,17,18]
    p = [1, 2, 3]
    param_grid = {'n_neighbors' : k, 'p' : p}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_
#






#####OPTIMIZE KNN
    
params = knn_param_selection(full_normPCA128_array[train_indicies],full_isAnimal_array[train_indicies],5)
#{'n_neighbors': 10, 'p': 2}
#{'n_neighbors': 12, 'p': 2}




KNN_model = KNeighborsClassifier(n_neighbors=12,p=2)
KNN_model.fit(full_normPCA128_array[train_indicies],full_isAnimal_array[train_indicies])
pred = KNN_model.predict(full_normPCA128_array[train_indicies])

is_correct(pred,full_isAnimal_array[test_indicies])









###########The same but fft first     DONT DO FFT


fftData = np.fft.fft(full_data_matrix)
fftData = np.abs(fftData)/np.size(full_isAnimal_array)
fft_normalized = preprocessing.scale(fftData)#normalize

pca = PCA(svd_solver='auto', n_components = 128)#PCA with all components
pca.fit(fft_normalized)
fft_normPCA_array = pca.transform(fft_normalized)


params = knn_param_selection(fft_normPCA_array[train_indicies],full_isAnimal_array[train_indicies],5)
#{'n_neighbors': 15, 'p': 1}



KNN_model = KNeighborsClassifier(n_neighbors=15,p=1)
KNN_model.fit(fft_normPCA_array[train_indicies],full_isAnimal_array[train_indicies])
pred = KNN_model.predict(fft_normPCA_array[train_indicies])

is_correct(pred,full_isAnimal_array[test_indicies])










svm_object = svm.SVC(C=1.0, gamma=2e-05,probability=True)
svm_object.fit(fft_normPCA_array[train_indicies],full_isAnimal_array[train_indicies])

svm_object.score(fft_normPCA_array[test_indicies],full_isAnimal_array[test_indicies])









