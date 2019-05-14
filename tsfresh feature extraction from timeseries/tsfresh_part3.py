# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:09:11 2019

@author: Ralle
"""

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
ntime = 576

extracted_features = np.zeros((nobs,788*64))

time = np.arange(ntime)

for i in range(nobs):
    print(i)
    ids = np.repeat(i,ntime)
    
    chanels_data = np.zeros((64,576))
    for j in range(64):
        chanels_data[j,:] = full_data_matrix[i,j*576:(j*576+576)]
    
    
    d = {"time":time,"id":ids,"ch0":chanels_data[0,:],"ch1":chanels_data[1,:],"ch2":chanels_data[2,:],"ch3":chanels_data[3,:],"ch4":chanels_data[4,:],"ch5":chanels_data[5,:],"ch6":chanels_data[6,:],"ch7":chanels_data[7,:],"ch8":chanels_data[8,:],"ch9":chanels_data[9,:],"ch10":chanels_data[10,:],"ch11":chanels_data[11,:],"ch12":chanels_data[12,:],"ch13":chanels_data[13,:],"ch14":chanels_data[14,:],"ch15":chanels_data[15,:],"ch16":chanels_data[16,:],"ch17":chanels_data[17,:],"ch18":chanels_data[18,:],"ch19":chanels_data[19,:],"ch20":chanels_data[20,:],"ch21":chanels_data[21,:],"ch22":chanels_data[22,:],"ch23":chanels_data[23,:],"ch24":chanels_data[24,:],"ch25":chanels_data[25,:],"ch26":chanels_data[26,:],"ch27":chanels_data[27,:],"ch28":chanels_data[28,:],"ch29":chanels_data[29,:],"ch30":chanels_data[30,:],"ch31":chanels_data[31,:],"ch32":chanels_data[32,:],"ch33":chanels_data[33,:],"ch34":chanels_data[34,:],"ch35":chanels_data[35,:],"ch36":chanels_data[36,:],"ch37":chanels_data[37,:],"ch38":chanels_data[38,:],"ch39":chanels_data[39,:],"ch40":chanels_data[40,:],"ch41":chanels_data[41,:],"ch42":chanels_data[42,:],"ch43":chanels_data[43,:],"ch44":chanels_data[44,:],"ch45":chanels_data[45,:],"ch46":chanels_data[46,:],"ch47":chanels_data[47,:],"ch48":chanels_data[48,:],"ch49":chanels_data[49,:],"ch50":chanels_data[50,:],"ch51":chanels_data[51,:],"ch52":chanels_data[52,:],"ch53":chanels_data[53,:],"ch54":chanels_data[54,:],"ch55":chanels_data[55,:],"ch56":chanels_data[56,:],"ch57":chanels_data[57,:],"ch58":chanels_data[58,:],"ch59":chanels_data[59,:],"ch60":chanels_data[60,:],"ch61":chanels_data[61,:],"ch62":chanels_data[62,:],"ch63":chanels_data[63,:]} 

    #d = {'ID': pd.Series(ids), 'time': pd.Series(time),'x': pd.Series(full_data_matrix[i])}
    df = pd.DataFrame(d)
    extracted_features[i,:] = extract_features(df, column_id="id", column_sort="time", default_fc_parameters=EfficientFCParameters())



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


os.chdir('C:/Users/Ralle/Documents/GitHub/AdvancedMachineLearning/tsfresh feature extraction from timeseries')
tsfresh_exstracted_from_concatChanels = np.load('tsfresh_exstracted_from_concatChanels.npy')
full_normalized_array = preprocessing.scale(tsfresh_exstracted_from_concatChanels)#normalize
############################ PCA_ 290 components holds 95% of variance
#pca = PCA(svd_solver='auto', n_components = 290)#PCA with all components
#pca.fit(full_normalized_array[train_indicies])
#full_normPCA_array = pca.transform(full_normalized_array)

from sklearn import svm


def svc_param_selection2(X, y, nfolds, Cs, gammas): #https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

Cs = [1e-2,1e-1,1e-0,1e+1,1e+2,1e+3,1e+4]
gammas = [1e-8,1e-7,1e-6, 1e-5,1e-4,1e-3,1e-2]
params = svc_param_selection2(full_normalized_array[train_indicies], full_subClass_array[train_indicies], 5, Cs, gammas)
#first run {'C': 10.0, 'gamma': 0.0001}#subclasses
# {'C': 10.0, 'gamma': 0.0001} binary
Cs = [2,4,6,8,9,10,20,30,50,70,90]
gammas = [2e-5,4e-5,6e-5,8e-5,9e-5, 1e-4,2e-4,3e-4,5e-4,7e-4,9e-4]
params = svc_param_selection2(full_normalized_array[train_indicies], full_subClass_array[train_indicies], 5, Cs, gammas)
#second run : {'C': 2, 'gamma': 0.0003} binary
#{'C': 8, 'gamma': 6e-05}#sub





svm_object = svm.SVC(C=8, gamma=6e-05)
svm_object.fit(full_normalized_array[train_indicies],full_subClass_array[train_indicies])

print(svm_object.score(full_normalized_array[test_indicies],full_subClass_array[test_indicies]))
predictions = svm_object.predict(full_normalized_array[test_indicies])

corr = is_correct(predictions,full_subClass_array[test_indicies])
N = np.size(full_subClass_array[test_indicies])
accuracy = np.sqrt(corr*(100-corr)/(100*N))


