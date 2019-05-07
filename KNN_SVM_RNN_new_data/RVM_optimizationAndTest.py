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
import matplotlib.pyplot as plt

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
full_normPCA600_array = np.load('full_normPCA600.npy')
full_normPCA123_array = np.load('full_normPCA123.npy')
#######################################################################


def rvc_param_selection(X, y, nfolds): #https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
    gammas = [1e-5]
    #gammas = [1e-8,1e-7,1e-6, 1e-5,1e-4,1e-3,1e-2]
    param_grid = {'coef1' : gammas}
    grid_search = GridSearchCV(RVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

#def rvc_param_selection2(X, y, nfolds): #https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
#    gammas = [1e-4,1e-3,1e-2]
#    param_grid = {'coef1' : gammas}
#    grid_search = GridSearchCV(RVC(kernel='rbf'), param_grid, cv=nfolds)
#    grid_search.fit(X, y)
#    grid_search.best_params_
#    return grid_search.best_params_




#WE START with the biary case
    
X = full_normPCA123_array[train_indicies]
Y = full_isAnimal_array[train_indicies]

params = rvc_param_selection(X,Y,5)
#
#params = rvc_param_selection2(X,Y,5)



####################TO TEST
test_err = np.zeros((15))
train_err = np.zeros((15))
coef = np.multiply([1000, 500, 200, 150, 100, 70, 60, 50, 40, 30, 20, 10, 5, 2, 1],1e-6)
for i in range(15):
    #coef[i] = 5e-5-(i+1)*1e-5
    clf = RVC(kernel="rbf",coef1=coef[i])# coef1:  1=46 0.1same 
    clf.fit(full_normPCA123_array[train_indicies], full_subClass_array[train_indicies])
    train_err[i] = clf.score(full_normPCA123_array[train_indicies],full_subClass_array[train_indicies])
    test_err[i] = clf.score(full_normPCA123_array[test_indicies],full_subClass_array[test_indicies])
    
    print(coef[i])
    print(train_err[i])
    print(test_err[i])
    print("\n\n")


#####################################################################################################################
rvm_tested_coef1 = np.load("rvm_tested_coef1.npy")
rvm_tested_coef1_again = np.load("rvm_tested_coef1_again.npy")
train1 = np.load("rvm_tested_coef1_trainerr.npy")
train2 = np.load("rvm_tested_coef1_trainerr_again.npy")
test1 = np.load("rvm_tested_coef1_testerr.npy")
test2 = np.load("rvm_tested_coef1_testerr_again.npy")


train1 = np.trim_zeros(train1)
train2 = np.trim_zeros(train2)
test1 = np.trim_zeros(test1)
test2 = np.trim_zeros(test2)

rvm_tested_coef1 = rvm_tested_coef1[0:np.size(train1)]
rvm_tested_coef2 = rvm_tested_coef1_again[0:np.size(train2)]

test_together = np.concatenate((test1,test2))
train_together = np.concatenate((train1,train2))
coef_together = np.concatenate((rvm_tested_coef1,rvm_tested_coef2))

plt.figure()
plt.scatter(coef_together,1-train_together)
plt.scatter(coef_together,1-test_together)
plt.title("RVM Error rate on single training and test set. Binary case")
plt.legend(['training error','test error'])
plt.xlabel("rbf kernel gamma")
plt.ylabel("classification error rate [%]")
plt.xlim(0,np.max(coef_together))
plt.ylim(0)
tick,dummy = plt.xticks()
plt.xticks(tick, rotation=45)
plt.grid()
plt.show()


#########OR FOR SUBCLASSES
rvm_tested_coef1 = np.load("rvm_subClass_gammas.npy")
train1 = np.load("rvm_subClass_trainerr.npy")
test1 = np.load("rvm_subClass_testerr.npy")

train1 = np.trim_zeros(train1)
test1 = np.trim_zeros(test1)

rvm_tested_coef1 = rvm_tested_coef1[0:np.size(train1)]


plt.figure()
plt.scatter(rvm_tested_coef1,1-train1)
plt.scatter(rvm_tested_coef1,1-test1)
plt.title("Classification on a single training and test set")
plt.legend(['training error','test error'])
plt.xlabel("rbf kernel gamma")
plt.ylabel("classification error rate [%]")
plt.xlim(0,np.max(rvm_tested_coef1))
plt.ylim(0)
tick,dummy = plt.xticks()
plt.xticks(tick, rotation=45)
plt.grid()
plt.show()
####################################




######################################################################################################################


###FOR MATLAB
#      scipy.io.savemat


clf = RVC(kernel="rbf",coef1=1e-5)# coef1:  1=46 0.1same 
clf.fit(full_normPCA123_array[train_indicies], full_isAnimal_array[train_indicies])
clf.score(full_normPCA123_array[train_indicies],full_isAnimal_array[train_indicies])
clf.score(full_normPCA123_array[test_indicies],full_isAnimal_array[test_indicies])


#predictions = clf.predict(full_normPCA128_array[test_indicies])
#
#corr=0
#for i in range(np.size(predictions)):
#    if predictions[i] == full_isAnimal_array[test_indicies][i]:
#        corr += 1
        