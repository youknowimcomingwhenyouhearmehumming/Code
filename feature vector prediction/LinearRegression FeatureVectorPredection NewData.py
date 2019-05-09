import scipy.io
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
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
pca = PCA(10, svd_solver='auto')
pca.fit(normal_data_all)
normal_data_pca = pca.transform(normal_data_all)#transform data to xx components



#####################################################################################
##We use the image_sematics from each to train after
n_observations =2160
X_train, X_test, y_train_index, y_test_index = train_test_split(normal_data_pca[range(n_observations),:],range(n_observations),test_size=0.2) #The reason why y-labels are not sorte least to largest is because this funktion mix things arround in order to get a mo

#feature_vector=np.zeros(2048,)


#y_train_index=np.array(y_train_index)
#y_train_index.astype(int)

#
#mean_err = np.zeros((1,1))
#for i in range(1):   
#    n_semantic_as_y = i #the 1st semantic is used as output
#    clf=LinearRegression()
#    clf.fit(X_train,full_semantics_matrix[y_train_index,n_semantic_as_y])
#    
#    predicted_out = clf.predict(X_test) #full_semantics_matrix[y_test_index,n_semantic_as_y]
#
#    ## calc error from test output
#    err = predicted_out-full_semantics_matrix[y_test_index,n_semantic_as_y]
#    mean_err[i] = np.mean(err)
#    print(i)


clf1=LinearRegression()
clf1.fit(X_train,full_semantics_matrix[y_train_index])
predicted_out = clf1.predict(X_test) #full_semantics_matrix[y_test_index,n_semantic_as_y]
err = predicted_out-full_semantics_matrix[y_test_index]
mean_err = np.mean(abs(err))
print('mean_err=',mean_err)


plt.figure()
plt.plot(mean_err)
plt.show()