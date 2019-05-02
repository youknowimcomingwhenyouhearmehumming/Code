# Importing pandas as pd 
import pandas as pd 
  
# Creating the first Dataframe using dictionary 
df1 = df = pd.DataFrame({"ID":[1, 2, 3, 4], 
                         "Time":[5, 6, 7, 8]}) 


## -*- coding: utf-8 -*-
#"""
#Created on Mon Apr 29 11:19:30 2019
#
#@author: Ralle
#"""
#import scipy.io
#import numpy as np
#import os
#import pickle
#from sklearn import svm
#from sklearn.model_selection import GridSearchCV
#
##For the test of comparing KNN/SVM/RVM basic
#
#
##First we get load the splits indicies and data
##To load:
##os.chdir('C:/Users/Bruger/Documents/Uni/Advanche machine learning/Projekt/Code/KNN_SVM_RNN_new_data')
#
#with open("train_indexes.txt", "rb") as fp:   # Unpickling
#    train_indicies = pickle.load(fp)
#with open("test_indexes.txt", "rb") as fp:   # Unpickling
#    test_indicies = pickle.load(fp)
#
#full_isAnimal_array = np.load('full_isAnimal_array.npy')
#full_subClass_array = np.load('full_subClass_array.npy')
#full_normPCA655_array = np.load('full_normPCA655_array.npy')
#full_normPCA128_array = np.load('full_normPCA128_array.npy')
########################################################################
#
##WE START with the biary case
##SVM
#def svc_param_selection(X, y, nfolds): #https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
#    Cs = [1e-2,1e-1,1e-0,1e+1,1e+2,1e+3,1e+4]
#    gammas = [1e-8,1e-7,1e-6, 1e-5,1e-4,1e-3,1e-2]
#    param_grid = {'C': Cs, 'gamma' : gammas}
#    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
#    grid_search.fit(X, y)
#    grid_search.best_params_
#    return grid_search.best_params_
#
#def svc_param_selection2(X, y, nfolds, Cs, gammas): #https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
#    param_grid = {'C': Cs, 'gamma' : gammas}
#    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
#    grid_search.fit(X, y)
#    grid_search.best_params_
#    return grid_search.best_params_
#
#
#X = full_normPCA128_array[train_indicies]
#Y = full_isAnimal_array[train_indicies]
#
#print('first')
##FIRST
##params1 = svc_param_selection(X, Y, 5)
##C1 = params1['C']
##gamma1 = params1['gamma']
##{'C': 100.0, 'gamma': 1e-06}
#
#print('second')
##SECOND
##Cs = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
##gammas = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
##Cs = [x*C1 for x in Cs]
##gammas = [x*gamma1 for x in gammas]
##params2 = svc_param_selection2(X, Y, 5, Cs, gammas)
##C2 = params2['C']
##gamma2 = params2['gamma']
##{'C': 20.0, 'gamma': 2e-06}
#
#print('third')
###THIRD
##Cs = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
##gammas = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
##Cs = [x*C2 for x in Cs]
##gammas = [x*gamma2 for x in gammas]
###params3 = svc_param_selection2(X, Y, 5, Cs, gammas)
##C3 = params3['C']
##gamma3 = params3['gamma']
###{'C':4 , 'gamma': 8e-6}
#
#print('fourth')
##fourth
##Cs = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
##gammas = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
##Cs = [x*C3 for x in Cs]
##gammas = [x*gamma3 for x in gammas]
##params4 = svc_param_selection2(X, Y, 5, Cs, gammas)
##C4 = params3['C']
##gamma4 = params3['gamma']
##{'C':4 , 'gamma': 8e-6}
#
#
######################TO TEST
##svm_object = svm.SVC(C=4.0, gamma=8e-06)
##svm_object.fit(full_normPCA128_array[train_indicies],full_isAnimal_array[train_indicies])
##
##svm_object.score(full_normPCA128_array[test_indicies],full_isAnimal_array[test_indicies])
##
##
##
##
##
##
#
#
#
#
#
#
#
#
###########################################################
################LETS DO ALL THE SAME WITH ALL SUBCLASSES:
#
#
#X = full_normPCA128_array[train_indicies]
#Y = full_subClass_array[train_indicies]
#
#
#
#print('first')
##FIRST
##params1 = svc_param_selection(X, Y, 5)
##C1 = params1['C']
##gamma1 = params1['gamma']
###{'C': 10.0, 'gamma': 1e-06}
#
#print('second')
##SECOND
##Cs = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
##gammas = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
##Cs = [x*C1 for x in Cs]
##gammas = [x*gamma1 for x in gammas]
##params2 = svc_param_selection2(X, Y, 5, Cs, gammas)
##C2 = params2['C']
##gamma2 = params2['gamma']
###{'C': 4.0, 'gamma': 6e-06}
#
#print('third')
##THIRD
##Cs = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
##gammas = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
##Cs = [x*C2 for x in Cs]
##gammas = [x*gamma2 for x in gammas]
###params3 = svc_param_selection2(X, Y, 5, Cs, gammas)
##C3 = params3['C']
##gamma3 = params3['gamma']
###{'C': 3.2, 'gamma': 6e-06}
#
#print('fourth')
##fourth
##Cs = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
##gammas = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
##Cs = [x*C3 for x in Cs]
##gammas = [x*gamma3 for x in gammas]
###params4 = svc_param_selection2(X, Y, 5, Cs, gammas)
##C4 = params3['C']
##gamma4 = params3['gamma']
###{'C': 3.2, 'gamma': 6e-06}
#
##
######################TO TEST
##svm_object = svm.SVC(C=3.2, gamma=6e-06)
##svm_object.fit(full_normPCA128_array[train_indicies],full_subClass_array[train_indicies])
##
##svm_object.score(full_normPCA128_array[test_indicies],full_subClass_array[test_indicies])
#
#
#
#
#
#
#for i in range(leng)
#
#label=1*full_subClass_array[test_indicies]=='airplane' and 2*full_subClass_array[test_indicies]=='elephant'
#print(label)
#
#
#
#
#
#
#
#
##For KNN:
##Do cross validation to find of classification error
#
#
##FOR SVM
##Optimize parameters on the training set
##Get error from test set
#
#
##FOR RVM
##Optimize parameters on the training set
##Get error from test set