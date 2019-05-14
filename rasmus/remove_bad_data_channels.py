# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:58:02 2019

@author: Ralle
"""

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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn import preprocessing
from skrvm import RVC

#For the test of comparing KNN/SVM/RVM basic


def is_correct(pred,y_test):# returns percent of correct predictions
    n = np.size(y_test)
    percent_correct = 0
    for i in range(n):
        if (y_test[i] == pred[i]):
            percent_correct += 1
    percent_correct = percent_correct/n
    return percent_correct


X = full_normPCA600_array[train_indicies]
Y = full_isAnimal_array[train_indicies]

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
test_window_size = 576# number of dimentions to leave out at a time
n_tests = int(np.floor(np.size(full_normalized_array[train_indicies,:],1)/test_window_size)) #make window size to match data size: must give an int
plt.axis([0, np.size(full_data_matrix,1), 0.3, 0.6])
all_scores = np.zeros((np.size(full_data_matrix,1)))
cv = StratifiedShuffleSplit(5,test_size = 1/5,random_state=0)
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
    scores = cross_val_score(svm_object, train_normPCA_array, full_subClass_array[train_indicies], cv=cv)
    score = np.mean(scores)
    all_scores[i*test_window_size:i*test_window_size+test_window_size] = [score]*test_window_size
    print('plotting')
    plt.scatter(np.add(np.arange(test_window_size),test_window_size*i),[score]*test_window_size)  
    plt.pause(0.05)
plt.show()








os.chdir('C:/Users/Ralle/Documents/GitHub/AdvancedMachineLearning/rasmus')
all_scores = np.load('all_scores_576size_bins.npy')

#####################################################
##Remove one bin at a time and test
cv = StratifiedShuffleSplit(5,test_size = 1/5,random_state=0)

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
#DO PCA 95%
    print('PCA')
    pca = PCA(svd_solver='auto', n_components = 123)#PCA with all components
    pca.fit(data_to_test)
    train_normPCA_array = pca.transform(data_to_test)
    
    svm_object = svm.SVC(C=3.2, gamma=6e-06)
    print('scoring')
    scores = cross_val_score(svm_object, train_normPCA_array, full_subClass_array[train_indicies], cv=cv)
    score = np.mean(scores)
    all_del_scores[i] = score
    print('plotting')
    plt.scatter(i,score)  
    plt.pause(0.05)
plt.show()
# Get the indices of maximum element in numpy array




#######################TRY PLOTTING WHICH DISSAPEARS

half_dims = dims_to_del[0:int(576*20)]
plt.figure()
for i in range(64):
    plt.subplot(64,1,i+1)
    x = half_dims[((half_dims < (i*576+576)) & (half_dims > (i*576)))]
    plt.scatter(np.mod(x,(i)*576),np.ones(np.size(x)))
    ax = plt.gca()
    ax.yaxis.set_visible(False)
    plt.axis([0,576,0.9,1.1])
    
    
    


##COMBINE THE REMOVED BAD DATA WITH SOME maximum PCA
raw_data_reduced_dim = np.delete(full_normalized_array,dims_to_del[0:int(576*20)],1)

####################################################################################DO PCA STUFF
################PCA AND VARIANCE EXPLAINED
pca = PCA(svd_solver='auto')#PCA with all components
pca.fit(raw_data_reduced_dim[train_indicies])
pca_cumsum = np.cumsum(pca.explained_variance_ratio_)*100

plt.figure()
plt.plot(pca_cumsum)
plt.grid()
plt.ylabel('Cummulative Variance Explained [%]')
plt.xlabel('# of Features')
plt.title('PCA Variance Explained')
plt.ylim(0,100.5)
#plt.plot(range(2160),np.repeat(95,2160))
#plt.legend(['Cumulative variance explained'])
plt.show()

############################ PCA_ 486 components holds 95% of variance using train set as base
pca = PCA(svd_solver='auto', n_components = 486)#PCA with all components
pca.fit(raw_data_reduced_dim[train_indicies])
full_normPCA486_array_reduced_dim = pca.transform(raw_data_reduced_dim)

###################################################################################find SVM coeffs
def svc_param_selection2(X, y, nfolds, Cs, gammas): #https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

Cs = [1e-2,1e-1,1e-0,1e+1,1e+2,1e+3,1e+4]
gammas = [1e-8,1e-7,1e-6, 1e-5,1e-4,1e-3,1e-2]
params = svc_param_selection2(full_normPCA486_array_reduced_dim[train_indicies], full_subClass_array[train_indicies], 5, Cs, gammas)
#first run {'C': 10.0, 'gamma': 1e-05}#subclasses
# {'C': 10.0, 'gamma': 1e-05} binary
Cs = [2,4,6,8,9,10,20,30,50,70,90]
gammas = [2e-6,4e-6,6e-6,8e-6,9e-6, 1e-5,2e-5,3e-5,5e-5,7e-5,9e-5]
params = svc_param_selection2(full_normPCA486_array_reduced_dim[train_indicies], full_subClass_array[train_indicies], 5, Cs, gammas)
#second run : {'C': 6, 'gamma': 1e-05}
#  {'C': 8, 'gamma': 1e-05}binary
Cs = [9.2, 9.4, 9.6, 9.8,10, 10.2, 10.4, 10.6, 10.8]
gammas = [4.4e-6,4.7e-6,5.2e-6,5.5e-6,5.8e-6, 6e-6,6.2e-6,6.5e-6,6.8e-6,7.3e-6,7.6e-6]
params = svc_param_selection2(full_normPCA486_array_reduced_dim[train_indicies], full_subClass_array[train_indicies], 5, Cs, gammas)
# third
# for binary
###################################################################################test on the test set
####################TO TEST
svm_object = svm.SVC(C=6, gamma=1e-5)
svm_object.fit(full_normPCA486_array_reduced_dim[train_indicies],full_subClass_array[train_indicies])

svm_object.score(full_normPCA486_array_reduced_dim[test_indicies],full_subClass_array[test_indicies])
predictions = svm_object.predict(full_normPCA486_array_reduced_dim[test_indicies])

corr = is_correct(predictions,full_subClass_array[test_indicies])
N = np.size(full_subClass_array[test_indicies])
accuracy = np.sqrt(corr*(100-corr)/(100*N))
#BINGO MAX




























dims_to_keep = []
chan_numbers = np.concatenate((np.add(np.arange(12),20),(np.add(np.arange(8),57))))
for i in range(np.size(chan_numbers)):
    dims_to_keep = np.concatenate((dims_to_keep,np.add((chan_numbers[i]-1)*576,np.arange(576))))

#######################TRY PLOTTING WHICH DISSAPEARS

plt.figure()
for i in range(64):
    plt.subplot(64,1,i+1)
    x = dims_to_remove_phys[((dims_to_keep < (i*576+576)) & (dims_to_keep >= (i*576)))]
    plt.scatter(np.mod(x,576),np.ones(np.size(x)))
    ax = plt.gca()
    ax.yaxis.set_visible(False)
    plt.axis([0,576,0.9,1.1])

dims_to_keep = dims_to_keep.astype(int)
data_reduced = full_normalized_array[:,dims_to_keep]


################PCA AND VARIANCE EXPLAINED
pca = PCA(svd_solver='auto')#PCA with all components
pca.fit(data_reduced[train_indicies])
pca_cumsum = np.cumsum(pca.explained_variance_ratio_)*100

plt.figure()
plt.plot(pca_cumsum)
plt.grid()
plt.ylabel('Cummulative Variance Explained [%]')
plt.xlabel('# of Features')
plt.title('PCA Variance Explained')
plt.ylim(0,100.5)
#plt.plot(range(2160),np.repeat(95,2160))
#plt.legend(['Cumulative variance explained'])
plt.show()

############################ PCA_ 301 components holds 95% of variance using train set as base
pca = PCA(svd_solver='auto', n_components = 301)#PCA with all components
pca.fit(data_reduced[train_indicies])
full_normPCA301_array_reduced_dim = pca.transform(data_reduced)





###################################################################################find SVM coeffs
def svc_param_selection2(X, y, nfolds, Cs, gammas): #https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

Cs = [1e-2,1e-1,1e-0,1e+1,1e+2,1e+3,1e+4]
gammas = [1e-8,1e-7,1e-6, 1e-5,1e-4,1e-3,1e-2]
params = svc_param_selection2(full_normPCA301_array_reduced_dim[train_indicies], full_isAnimal_array[train_indicies], 5, Cs, gammas)
#first run {'C': 10000.0, 'gamma': 1e-08}#subclasses
#   {'C': 10.0, 'gamma': 0.0001} binary
Cs = [2,4,6,8,10,20,30,40,60,80]
gammas = [2e-5,4e-5,6e-5,8e-5,1e-4,2e-4,3e-4,4e-4,6e-4,8e-4]
params = svc_param_selection2(full_normPCA301_array_reduced_dim[train_indicies], full_isAnimal_array[train_indicies], 5, Cs, gammas)
#second run : {'C': 6000.0, 'gamma': 8e-09} subclasses

Cs = [2e+3, 4e+3, 6e+3, 8e+3, 1e+4,3e+4, 5e+4,1e+5,1e+6]
gammas = [1e-9, 3e-9, 6e-9, 8e-9, 1e-8, 3e-8, 5e-8, 8e-8]
params = svc_param_selection2(full_normPCA301_array_reduced_dim[train_indicies], full_isAnimal_array[train_indicies], 5, Cs, gammas)
#second run : {'C': 6000.0, 'gamma': 8e-09} subclasses

####################TO TEST
svm_object = svm.SVC(C=6000, gamma=8e-9)
svm_object.fit(full_normPCA301_array_reduced_dim[train_indicies],full_isAnimal_array[train_indicies])

svm_object.score(full_normPCA301_array_reduced_dim[test_indicies],full_isAnimal_array[test_indicies])
predictions = svm_object.predict(full_normPCA301_array_reduced_dim[test_indicies])

corr = is_correct(predictions,full_isAnimal_array[test_indicies])
N = np.size(full_isAnimal_array[test_indicies])
accuracy = np.sqrt(corr*(100-corr)/(100*N))
#BINGO MAX






