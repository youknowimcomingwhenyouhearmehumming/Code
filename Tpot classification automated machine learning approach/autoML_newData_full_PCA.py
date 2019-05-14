import scipy.io
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import os
from sklearn.metrics import classification_report,confusion_matrix,precision_score

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

#
#def is_animal(class_vector):#creates vector of 1 if animal and 0 if not
#    n = np.size(class_vector)
#    bin_vector = np.zeros(n)
#    for i in range(n):
#        if class_vector[i] == 'animal':
#            bin_vector[i] = 1
#        else:
#            bin_vector[i] = 0
#    return bin_vector



os.chdir('C:/Users/Bruger/Documents/Uni/Advanche machine learning/Projekt/new_data/data')


n_experiments = 4


full_data_matrix = []
full_superClass_array = []
full_subClass_array = []
full_semantics_matrix = []

for i in range(n_experiments):
    eeg_events = scipy.io.loadmat('exp' + str(i+1) + '\eeg_events.mat')
    image_order = np.genfromtxt('exp' + str(i+1) + '\image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#
    data = eeg_events["eeg_events"]
    concat_data = concat_channels(data)
    #load sematics
    image_semantics_mat = scipy.io.loadmat('exp' + str(i+1) + '\image_semantics.mat')
    image_semantics = image_semantics_mat["image_semantics"]#semantics_vector x n_images
    
    if i == 0:
        full_data_matrix = concat_data
        full_superClass_array = image_order[:,0]
        full_subClass_array = image_order[:,1]
        full_semantics_matrix = np.transpose(image_semantics)

    else:
        full_data_matrix  = np.concatenate((full_data_matrix,concat_data),axis=0)
        full_superClass_array  = np.concatenate((full_superClass_array,image_order[:,0]),axis=0)
        full_subClass_array  = np.concatenate((full_subClass_array,image_order[:,1]),axis=0)
        full_semantics_matrix = np.concatenate((full_semantics_matrix,np.transpose(image_semantics)))
        
#Normalize data
  
normal_data_all = preprocessing.scale(full_data_matrix)#normalize



#Change superclass toAnimal or not:
#full_isAnimal_array = is_animal(full_superClass_array)


"""
PCA stuff method 1
"""
pca = PCA(2160, svd_solver='auto')
pca.fit(normal_data_all)
normal_data_pca = pca.transform(normal_data_all)#transform data to xx components

n_observations =2160
X_train, X_test, y_train_index, y_test_index = train_test_split(normal_data_pca[range(n_observations),:],range(n_observations),test_size=0.2) #The reason why y-labels are not sorte least to largest is because this funktion mix things arround in order to get a mo

y_train=full_subClass_array[y_train_index]

y_test=full_subClass_array[y_test_index]


from tpot import TPOTClassifier
clf=TPOTClassifier(verbosity=2,n_jobs=1)
clf.fit(X_train,y_train)

print('test score=',clf.score(X_test,y_test))
predictions = clf.predict(X_test)
print(confusion_matrix(y_test,predictions))

