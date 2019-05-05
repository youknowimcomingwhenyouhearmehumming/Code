# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:55:19 2019

@author: Ralle
"""

import scipy.io
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle

import pandas as pd 
from tsfresh import extract_relevant_features
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

#full_data_matrix
#full_subClass_array
#full_isAnimal_array



#oneD_data = np.reshape(full_data_matrix,(2160*36864,))
#
#ids = np.zeros((np.size(oneD_data)))
#time = np.zeros((np.size(oneD_data)))
#for i in range(np.size(full_data_matrix,0)):
#    #ids[i*np.size(full_data_matrix,1):np.size(full_data_matrix,1)+i*np.size(full_data_matrix,1)] = np.repeat(i,np.size(full_data_matrix,1))
#    #time[i*np.size(full_data_matrix,1):np.size(full_data_matrix,1)+i*np.size(full_data_matrix,1)] = np.arange(np.size(full_data_matrix,1))
#    ids[i*100:100+i*100] = np.repeat(i,np.size(full_data_matrix,1))
#    time[i*np.size(full_data_matrix,1):np.size(full_data_matrix,1)+i*np.size(full_data_matrix,1)] = np.arange(np.size(full_data_matrix,1))
#
#d = {'ID': pd.Series(ids), 'time': pd.Series(time),'x': pd.Series(oneD_data)}
#
#df = pd.DataFrame(d)
#
#extracted_features = extract_features(df, column_id="ID", column_sort="time")


nobs = np.size(full_data_matrix,0)
ntime = np.size(full_data_matrix,1)

extracted_features = np.zeros((nobs,788))

time = np.arange(ntime)

for i in range(nobs):
    print(i)
    ids = np.repeat(i,ntime)
    
    d = {'ID': pd.Series(ids), 'time': pd.Series(time),'x': pd.Series(full_data_matrix[i])}
    df = pd.DataFrame(d)
    extracted_features[i,:] = extract_features(df, column_id="ID", column_sort="time", default_fc_parameters=EfficientFCParameters())



################################################Quick test
#Normalize data
full_normalized_array = preprocessing.scale(extracted_features)#normalize

################PCA AND VARIANCE EXPLAINED
pca = PCA(svd_solver='auto')#PCA with all components
pca.fit(full_normalized_array)
pca_cumsum = np.cumsum(pca.explained_variance_ratio_)*100

plt.figure()
plt.plot(pca_cumsum)
plt.grid()
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis of feature extracted data')
plt.ylim(0,100.5)
plt.plot(range(2160),np.repeat(95,2160))
plt.legend(['Cumulative variance explained','95%'])
plt.show()


############################ PCA_ 290 components holds 95% of variance
pca = PCA(svd_solver='auto', n_components = 290)#PCA with all components
pca.fit(full_normalized_array[train_indicies])
full_normPCA_array = pca.transform(full_normalized_array)

from sklearn import svm

svm_object = svm.SVC(C=10, gamma=1e-03)
svm_object.fit(extracted_features[train_indicies],full_isAnimal_array[train_indicies])

print(svm_object.score(extracted_features[test_indicies],full_isAnimal_array[test_indicies]))
predictions = svm_object.predict(extracted_features[test_indicies])





