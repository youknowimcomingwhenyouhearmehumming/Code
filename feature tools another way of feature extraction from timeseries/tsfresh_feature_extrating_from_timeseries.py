# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:08:49 2019

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


def concat_channels(eeg_events):#channels*EEG_value*img 
    #for putting all channels in a single vector. return is n_img x 17600
    n_channels,n_samples,n_img = np.shape(eeg_events)
    concat_all = np.zeros((n_img,n_channels*n_samples))
    for i in range(n_img): #for each image
        concat_row = []
        for j in range(n_channels): #for each channel of channels
            #concat_data[i] = np.concatenate((concat_data[i],eeg_events[j,:,i]),axis=1)
            concat_row = np.concatenate((concat_row,eeg_events[j,:,i]))
        concat_all[i] = concat_row
    return concat_all #[n_img*17600]


def is_animal(class_vector):#creates vector of 1 if animal and 0 if not
    n = np.size(class_vector)
    bin_vector = np.zeros(n)
    for i in range(n):
        if class_vector[i] == 'animal':
            bin_vector[i] = 1
        else:
            bin_vector[i] = 0
    return bin_vector



os.chdir('C:/Users/Bruger/Desktop/usb/Projekt/new_data/data')

n_experiments = 4


full_data_matrix = []
full_superClass_array = []
full_subClass_array = []
for i in range(n_experiments):
    eeg_events = scipy.io.loadmat('exp' + str(i+1) + '\eeg_events.mat')
    image_order = np.genfromtxt('exp' + str(i+1) + '\image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#
    data = eeg_events["eeg_events"]
    
    
    
    
    
    
    concat_data=data.reshape((data.shape[1]*data.shape[2]),data.shape[0]) #DETTE ER MIN KODE

#    concat_data = concat_channels(data)
    if i == 0:
        full_data_matrix = concat_data
        full_superClass_array = image_order[:,0]
        full_subClass_array = image_order[:,1]
    else:
        full_data_matrix  = np.concatenate((full_data_matrix,concat_data),axis=0)
        full_superClass_array  = np.concatenate((full_superClass_array,image_order[:,0]),axis=0)
        full_subClass_array  = np.concatenate((full_subClass_array,image_order[:,1]),axis=0)

#Change superclass toAnimal or not:
full_isAnimal_array = is_animal(full_superClass_array)



##Normalize data
#full_normalized_array = preprocessing.scale(full_data_matrix)#normalize

import numpy as np
ID=np.zeros(np.shape(full_data_matrix)[0])
time=np.zeros(np.shape(full_data_matrix)[0])
ID_number=1
time_number=0
for i in range(np.shape(full_data_matrix)[0]):
    if i%576==0:
        ID_number=ID_number+1
#        print(ID_number)
    if i%576==0:
        time_number=0
    ID[i]=ID_number
    time[i]=time_number
    time_number=time_number+1
ID=ID-2
A=ID


total=np.zeros([1244160,66])
total[:,0]=ID
total[:,1]=time
total[:,2:]=full_data_matrix

#total=np.concatenate(np.transpose(ID),np.transpose(time),full_data_matrix)
#
#import pandas as pd
#s = pd.Series(map(lambda x:[x], total)).apply(lambda x:x[0])

#from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
#    load_robot_execution_failures
#download_robot_execution_failures()
#timeseries, y = load_robot_execution_failures()

import pandas as pd 

y=pd.Series(full_isAnimal_array)
#
#for i in range(len(full_isAnimal_array)):
#    y[i]=full_isAnimal_array[i]
    



## Importing pandas as pd 
#  
# Creating the first Dataframe using dictionary 
df = pd.DataFrame({"time":time,"id":ID,"ch0":full_data_matrix[:,0],"ch1":full_data_matrix[:,1],"ch2":full_data_matrix[:,2],"ch3":full_data_matrix[:,3]}) 

#df2 = pd.DataFrame({"id":[ID]}) 
#df3 = pd.DataFrame(full_data_matrix) 
#
#A=df1.join(df2)
#B=A.join(df3)

from tsfresh import extract_relevant_features

features_filtered_direct = extract_relevant_features(df, y,column_id='id', column_sort='time', n_jobs=4)








"""
PCA stuff
"""

#################PCA AND VARIANCE EXPLAINED
#pca = PCA(svd_solver='auto')#PCA with all components
#pca.fit(full_normalized_array)
#pca_cumsum = np.cumsum(pca.explained_variance_ratio_)*100
#
#plt.figure()
#plt.plot(pca_cumsum)
#plt.grid()
#plt.ylabel('% Variance Explained')
#plt.xlabel('# of Features')
#plt.title('PCA Analysis')
#plt.ylim(0,100.5)
#plt.plot(range(2160),np.repeat(95,2160))
#plt.legend(['Cumulative variance explained','95%'])
#plt.show()
#
############################# PCA_ 655 components holds 95% of variance
#pca = PCA(svd_solver='auto', n_components = 655)#PCA with all components
#pca.fit(full_normalized_array)
#full_normPCA_array = pca.transform(full_normalized_array)
#
############################# PCA_ 128 components holds 75% of variance
#pca = PCA(svd_solver='auto', n_components = 128)#PCA with all components
#pca.fit(full_normalized_array)
#full_normPCA_array = pca.transform(full_normalized_array)
#
#
############################# CREATE TEST AND TRAINING FOR THE MODELS
##X_train, X_test, train_index, test_index = train_test_split(full_normPCA_array,range(np.size(full_isAnimal_array)),test_size=0.20)
#
#os.chdir('C:/Users/Ralle/Documents/GitHub/AdvancedMachineLearning/KNN_SVM_RNN_new_data')
#
##with open("train_indexes.txt", "wb") as fp:   #Pickling
##    pickle.dump(train_index, fp)
#
##with open("test_indexes.txt", "wb") as fp:   #Pickling
##    pickle.dump(test_index, fp)
#
#
##To load:
##with open("train_indexes.txt", "rb") as fp:   # Unpickling
##    b = pickle.load(fp)
#
