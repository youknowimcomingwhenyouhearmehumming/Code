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

#full_data_matrix
#full_subClass_array
#full_isAnimal_array



oneD_data = np.reshape(full_data_matrix,(2160*36864,))

ids = np.zeros((np.size(oneD_data)))
time = np.zeros((np.size(oneD_data)))
for i in range(np.size(full_data_matrix,0)):
    #ids[i*np.size(full_data_matrix,1):np.size(full_data_matrix,1)+i*np.size(full_data_matrix,1)] = np.repeat(i,np.size(full_data_matrix,1))
    #time[i*np.size(full_data_matrix,1):np.size(full_data_matrix,1)+i*np.size(full_data_matrix,1)] = np.arange(np.size(full_data_matrix,1))
    ids[i*100:100+i*100] = np.repeat(i,np.size(full_data_matrix,1))
    time[i*np.size(full_data_matrix,1):np.size(full_data_matrix,1)+i*np.size(full_data_matrix,1)] = np.arange(np.size(full_data_matrix,1))

d = {'ID': pd.Series(ids), 'time': pd.Series(time),'x': pd.Series(oneD_data)}

df = pd.DataFrame(d)

extracted_features = extract_features(df, column_id="ID", column_sort="time")




ids = np.repeat(1,np.size(full_data_matrix,1))
time = np.arange(np.size(full_data_matrix,1))

d = {'ID': pd.Series(ids), 'time': pd.Series(time),'x': pd.Series(full_data_matrix[1])}
df = pd.DataFrame(d)
extracted_features = extract_features(df, column_id="ID", column_sort="time")
