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
from skrvm import RVC

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
test_window_size = 144# number of dimentions to leave out at a time
n_tests = int(np.floor(np.size(full_normalized_array[train_indicies,:],1)/test_window_size)) #make window size to match data size: must give an int
plt.axis([0, np.size(full_data_matrix,1), 0.3, 0.6])
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








os.chdir('C:/Users/Ralle/Documents/GitHub/AdvancedMachineLearning/rasmus')
all_scores = np.load('all_scores_144size_bins.npy')

#####################################################
##Remove one bin at a time and test
plt.figure()
plt.axis([0,n_tests, 0.3, 0.6])
plt.grid()
dims_to_del = []
all_del_scores = np.zeros((n_tests))
data_to_test = full_normalized_array[train_indicies,:]
scores_temp_array = all_scores

for i in range(n_tests):
    #The highest scores is the least nessesary bins
    dims_to_del = np.concatenate((dims_to_del,np.where(scores_temp_array == np.amax(scores_temp_array))[0]))
    data_to_test = np.delete(full_normalized_array[train_indicies,:],dims_to_del,1)
    dims_to_del = dims_to_del.astype(int)
    scores_temp_array[dims_to_del] = 0
#DO PCA
    print('PCA')
    pca = PCA(svd_solver='auto', n_components = 123)#PCA with all components
    pca.fit(data_to_test)
    train_normPCA_array = pca.transform(data_to_test)
    
    svm_object = svm.SVC(C=3.2, gamma=6e-06)
    print('scoring')
    scores = cross_val_score(svm_object, train_normPCA_array, full_subClass_array[train_indicies], cv=3)
    score = np.mean(scores)
    all_del_scores[i] = score
    print('plotting')
    plt.scatter(i,score)  
    plt.pause(0.05)
plt.show()
# Get the indices of maximum element in numpy array








#
#
########################TRY PLOTTING WHICH DISSAPEARS
#window_size=144;
#n_test = 36864/window_size;
#n_to_remove = int(n_test/4)###############WRONG THE WHOLE SECTION
#dims_to_del = []
#scores_temp_array = all_scores
#
#for i in range(n_to_remove):
#    dims_to_del = np.concatenate((dims_to_del,np.where(scores_temp_array == np.amax(scores_temp_array))[0]))
#    scores_temp_array = np.delete(all_scores,dims_to_del)
#
#removed = np.ones((36864))
#dims_to_del=dims_to_del.astype(int)
#removed[dims_to_del] = 0
#
#plt.figure()
#for i in range(64):
#    plt.subplot(64,1,i+1)
#    plt.plot(removed[np.add(np.arange(576),i*576)])
#    #plt.scatter(dims_to_del,np.ones(np.size(dims_to_del)),'red')
#    ax = plt.gca()
#    ax.yaxis.set_visible(False)
#    
    
    
    


##COMBINE THE REMOVED BAD DATA WITH SOME maximum PCA
raw_data_reduced_dim = np.delete(full_normalized_array[train_indicies,:],dims_to_del[0:36864/2],1)

#DO PCA STUFF

#find SVM coeffs

#test on the test set

#BINGO MAX




    