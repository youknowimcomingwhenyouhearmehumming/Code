# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:30:15 2019

@author: Ralle
"""



import scipy.io
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.decomposition import PCA
import os
from sklearn.metrics import confusion_matrix





def is_correct(pred,y_test):# returns percent of correct predictions
    n = np.size(y_test)
    percent_correct = 0
    for i in range(n):
        if (y_test[i] == pred[i]):
            percent_correct += 1
    percent_correct = percent_correct/n
    return percent_correct





# 
n_models = 19
E_gen_s = np.zeros((n_models,1))
kf = KFold(n_splits=10, shuffle = True)


kf_inner = KFold(n_splits=10, shuffle = True)#Maybe shuffle on # one subject left out
kf_inner.get_n_splits(full_normPCA123_array[train_indicies])
for train_index_inner, test_index_inner in kf.split(full_normPCA123_array[train_indicies]): #inner split. one subject left out
    print('\t inner loop started')
    inner_trainX = full_normPCA123_array[train_indicies][train_index_inner,:]
    inner_testX = full_normPCA123_array[train_indicies][test_index_inner,:]
    inner_trainY = full_isAnimal_array[train_indicies][train_index_inner]
    inner_testY = full_isAnimal_array[train_indicies][test_index_inner]
    nDval = np.size(inner_testY)
    nDpar = np.size(full_isAnimal_array[train_indicies])
    for i in range(n_models):#number of models 1 to 10 neighbours
            #print('\t \t training models loop started')
            n_neigh = i+1
            #train
            KNN_model = KNeighborsClassifier(n_neighbors=n_neigh,p=1)#p is mahalanobis distance parameter
            KNN_model.fit(inner_trainX,inner_trainY) 
            #validate
            pred = KNN_model.predict(inner_testX)
            err = 1-is_correct(pred,inner_testY)
            print(err)
            E_gen_s[i] += err*nDval/nDpar
    n_opt = E_gen_s.argmin()+1
    print(n_opt)
    
    
KNN_model = KNeighborsClassifier(n_neighbors=9)
KNN_model.fit(full_normPCA123_array[train_indicies],full_isAnimal_array[train_indicies])
pred = KNN_model.predict(full_normPCA123_array[test_indicies])
corr = is_correct(pred,full_isAnimal_array[test_indicies])


N = np.size(full_isAnimal_array[test_indicies])
accuracy = np.sqrt(corr*(100-corr)/(100*N))



#################Creating the most beatiful graphs
classes = np.unique(full_subClass_array)
confu_matrix = np.zeros((2,np.size(classes)))
for i in range(np.size(classes)):
    for j in range(np.size(pred)):
        if full_subClass_array[test_indicies[j]] == classes[i]:
            if pred[j] == full_isAnimal_array[test_indicies[j]]: #TRUE
                confu_matrix[0,i]+=1
            else:                                                       #FALSE
                confu_matrix[1,i]+=1

confu_matrix_percent = np.zeros(np.shape(confu_matrix))
for i in range(np.size(classes)):
    for j in range(2):
        confu_matrix_percent[j,i] = confu_matrix[j,i]/np.sum(confu_matrix[:,i])*100

y_ax = ["True","False"]
x_ax = classes

fig, ax = plt.subplots()
im = ax.imshow(confu_matrix_percent)

ax.set_xticks(np.arange(len(x_ax)))
ax.set_yticks(np.arange(len(y_ax)))

ax.set_xticklabels(x_ax)
ax.set_yticklabels(y_ax)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(y_ax)):
    for j in range(len(x_ax)):
        text = ax.text(j, i, str(np.round(confu_matrix_percent[i,j],1)) + " %", ha="center", va="center", color="black")

ax.set_title("Predictet binary class compared to true class of test set using KNN")
fig.tight_layout()
plt.xlabel("True class")
plt.ylabel("Predicted binary class")
plt.show()








#################WITH ALL CLASSES














    
    
    
# 
n_models = 19
E_gen_s = np.zeros((n_models,1))
kf = KFold(n_splits=10, shuffle = True)


kf_inner = KFold(n_splits=10, shuffle = True)#Maybe shuffle on # one subject left out
kf_inner.get_n_splits(full_normPCA123_array[train_indicies])
for train_index_inner, test_index_inner in kf.split(full_normPCA123_array[train_indicies]): #inner split. one subject left out
    print('\t inner loop started')
    inner_trainX = full_normPCA123_array[train_indicies][train_index_inner,:]
    inner_testX = full_normPCA123_array[train_indicies][test_index_inner,:]
    inner_trainY = full_subClass_array[train_indicies][train_index_inner]
    inner_testY = full_subClass_array[train_indicies][test_index_inner]
    nDval = np.size(inner_testY)
    nDpar = np.size(full_subClass_array[train_indicies])
    for i in range(n_models):#number of models 1 to 10 neighbours
            #print('\t \t training models loop started')
            n_neigh = i+1
            #train
            KNN_model = KNeighborsClassifier(n_neighbors=n_neigh,p=1)#p is mahalanobis distance parameter
            KNN_model.fit(inner_trainX,inner_trainY) 
            #validate
            pred = KNN_model.predict(inner_testX)
            err = 1-is_correct(pred,inner_testY)
            print(err)
            E_gen_s[i] += err*nDval/nDpar
    n_opt = E_gen_s.argmin()+1
    print(n_opt)
    
   
    
KNN_model = KNeighborsClassifier(n_neighbors=18)
KNN_model.fit(full_normPCA123_array[train_indicies],full_subClass_array[train_indicies])
pred = KNN_model.predict(full_normPCA123_array[test_indicies])
corr = is_correct(pred,full_subClass_array[test_indicies])


N = np.size(full_subClass_array[test_indicies])
accuracy = np.sqrt(corr*(100-corr)/(100*N))

#################Creating the most beatiful graphs
classes = np.unique(full_subClass_array)
confu_matrix = np.zeros((np.size(classes),np.size(classes)))
for i in range(np.size(classes)):
    for j in range(np.size(pred)):
        if full_subClass_array[test_indicies[j]] == classes[i]:
            if classes[0] == pred[j]:
                confu_matrix[0,i]+=1
            elif classes[1] == pred[j]:
                confu_matrix[1,i]+=1
            elif classes[2] == pred[j]:
                confu_matrix[2,i]+=1
            elif classes[3] == pred[j]:
                confu_matrix[3,i]+=1
            elif classes[4] == pred[j]:
                confu_matrix[4,i]+=1
            elif classes[5] == pred[j]:
                confu_matrix[5,i]+=1

confu_matrix_percent = np.zeros(np.shape(confu_matrix))
for i in range(np.size(classes)):
    for j in range(np.size(classes)):
        confu_matrix_percent[j,i] = confu_matrix[j,i]/np.sum(confu_matrix[:,i])*100

y_ax = classes
x_ax = classes

fig, ax = plt.subplots()
im = ax.imshow(confu_matrix_percent)

ax.set_xticks(np.arange(len(x_ax)))
ax.set_yticks(np.arange(len(y_ax)))

ax.set_xticklabels(x_ax)
ax.set_yticklabels(y_ax)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(y_ax)):
    for j in range(len(x_ax)):
        text = ax.text(j, i, str(np.round(confu_matrix_percent[i,j],1)) + " %", ha="center", va="center", color="black")

ax.set_title("Predictet class compared to true class of test set using KNN")
fig.tight_layout()
plt.xlabel("True class")
plt.ylabel("Predicted class")
plt.show()


