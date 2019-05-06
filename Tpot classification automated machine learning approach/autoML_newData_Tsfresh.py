import scipy.io
import numpy as np
import os
import pickle
from sklearn import svm
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report,confusion_matrix,precision_score

from sklearn.decomposition import PCA
from sklearn import preprocessing
import scipy.io as sio
plt.close('all')

#For the test of comparing KNN/SVM/RVM basic


#First we get load the splits indicies and data

Tsfresh_data=np.load('tsfresh_exstracted_from_concatChanels.npy')

#To load:
os.chdir('C:/Users/Bruger/Documents/Uni/Advanche machine learning/Projekt/Code/KNN_SVM_RNN_new_data')

with open("train_indexes.txt", "rb") as fp:   # Unpickling
    train_indicies = pickle.load(fp)
with open("test_indexes.txt", "rb") as fp:   # Unpickling
    test_indicies = pickle.load(fp)

full_isAnimal_array = np.load('full_isAnimal_array.npy')
full_subClass_array = np.load('full_subClass_array.npy')
#full_normPCA655_array = np.load('full_normPCA655_array.npy')
#full_normPCA128_array = np.load('full_normPCA128_array.npy')



#######################################################################


#X = full_normPCA128_array[train_indicies]
#Y = full_isAnimal_array[train_indicies]
#
#X = full_normPCA128_array[train_indicies]
#Y = full_subClass_array[train_indicies]

"""
For binary
"""
#X_train=Tsfresh_data[train_indicies]
#y_train=full_isAnimal_array[train_indicies]
#
#X_test=Tsfresh_data[test_indicies]
#y_test=full_isAnimal_array[test_indicies]
#
#print(sum(y_test==1))
#print(len(y_test))
#print('percentage of not animals=',(sum(y_test==1)-len(y_test))/len(y_test))
#

"""
For all classes
"""




X_train=Tsfresh_data[train_indicies]
y_train_string=full_subClass_array[train_indicies]

X_test=Tsfresh_data[test_indicies]
y_test_string=full_subClass_array[test_indicies]

y_train=np.zeros(len(y_train_string))
y_test=np.ceil(np.zeros(len(y_test_string)))

for i in range(len(y_train_string)):
    if y_train_string[i]=='airplane':
        y_train[i]=1
    if y_train_string[i]=='elephant':
        y_train[i]=2
    if y_train_string[i]=='pizza':
        y_train[i]=3
    if y_train_string[i]=='sheep':
        y_train[i]=4
    if y_train_string[i]=='train':
        y_train[i]=5
    if y_train_string[i]=='zebra':
        y_train[i]=6

for i in range(len(y_test_string)):
    if y_test_string[i]=='airplane':
        y_test[i]=1
    if y_test_string[i]=='elephant':
        y_test[i]=2
    if y_test_string[i]=='pizza':
        y_test[i]=3
    if y_test_string[i]=='sheep':
        y_test[i]=4
    if y_test_string[i]=='train':
        y_test[i]=5
    if y_test_string[i]=='zebra':
        y_test[i]=6




#
"""
-------------- TPOT does is magic-------------------------------------
"""

from tpot import TPOTClassifier
clf=TPOTClassifier(verbosity=2,n_jobs=3, config_dict='TPOT light')
clf.fit(X_train,y_train)

print('test score=',clf.score(X_test,y_test))
predictions = clf.predict(X_test)
print(confusion_matrix(y_test,predictions))

