# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:19:30 2019

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
full_normPCA600_array = np.load('full_normPCA600.npy')
full_normPCA123_array = np.load('full_normPCA123.npy')




#######################################################################

#WE START with the biary case
#SVM
def svc_param_selection(X, y, nfolds): #https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
    Cs = [1e-2,1e-1,1e-0,1e+1,1e+2,1e+3,1e+4]
    gammas = [1e-8,1e-7,1e-6, 1e-5,1e-4,1e-3,1e-2]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

def svc_param_selection2(X, y, nfolds, Cs, gammas): #https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

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

print('first')
#FIRST
params1 = svc_param_selection(X, Y, 5)
C1 = params1['C']
gamma1 = params1['gamma']
#{'C': 100.0, 'gamma': 1e-06}
#for 600pca {'C': 100.0, 'gamma': 1e-05}

print('second')
#SECOND
Cs = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
gammas = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
Cs = [x*C1 for x in Cs]
gammas = [x*gamma1 for x in gammas]
params2 = svc_param_selection2(X, Y, 5, Cs, gammas)
C2 = params2['C']
gamma2 = params2['gamma']
#{'C': 20.0, 'gamma': 2e-06}
#{'C': 20.0, 'gamma': 2e-05} with 600 PCA

print('third')
#THIRD
Cs = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
gammas = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
Cs = [x*C2 for x in Cs]
gammas = [x*gamma2 for x in gammas]
params3 = svc_param_selection2(X, Y, 5, Cs, gammas)
C3 = params3['C']
gamma3 = params3['gamma']
#{'C':4 , 'gamma': 8e-6}
#{'C': 4.0, 'gamma': 1.6e-05} for 600pca

print('fourth')
#fourth
Cs = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
gammas = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
Cs = [x*C3 for x in Cs]
gammas = [x*gamma3 for x in gammas]
params4 = svc_param_selection2(X, Y, 5, Cs, gammas)
C4 = params3['C']
gamma4 = params3['gamma']
#{'C':4 , 'gamma': 8e-6}


####################TO TEST
svm_object = svm.SVC(C=4.0, gamma=1.6e-05,probability=True)
svm_object.fit(full_normPCA600_array[train_indicies],full_isAnimal_array[train_indicies])

svm_object.score(full_normPCA600_array[test_indicies],full_isAnimal_array[test_indicies])
predictions = svm_object.predict(full_normPCA600_array[test_indicies])
#WHEN probability = True it uses Platt scaling to estimate posterior distributions. this may be bad


corr = is_correct(predictions,full_isAnimal_array[test_indicies])
N = np.size(full_isAnimal_array[test_indicies])
accuracy = np.sqrt(corr*(100-corr)/(100*N))


#################Creating the most beatiful graphs
#################Creating the most beatiful graphs
pred = predictions
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

ax.set_title("Predictet binary class compared to true class of test set using SVM")
fig.tight_layout()
plt.xlabel("True class")
plt.ylabel("Predicted binary class")
plt.show()





##########################################################
###############LETS DO ALL THE SAME WITH ALL SUBCLASSES:


X = full_normPCA600_array[train_indicies]
Y = full_subClass_array[train_indicies]



print('first')
#FIRST
params1 = svc_param_selection(X, Y, 5)
C1 = params1['C']
gamma1 = params1['gamma']
#{'C': 10.0, 'gamma': 1e-06}
#{'C': 100.0, 'gamma': 1e-05} for 600
print('second')
#SECOND
Cs = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
gammas = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
Cs = [x*C1 for x in Cs]
gammas = [x*gamma1 for x in gammas]
params2 = svc_param_selection2(X, Y, 5, Cs, gammas)
C2 = params2['C']
gamma2 = params2['gamma']
#{'C': 4.0, 'gamma': 6e-06}
#{'C': 20.0, 'gamma': 2e-05} for 600pca
print('third')
#THIRD
Cs = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
gammas = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
Cs = [x*C2 for x in Cs]
gammas = [x*gamma2 for x in gammas]
params3 = svc_param_selection2(X, Y, 5, Cs, gammas)
C3 = params3['C']
gamma3 = params3['gamma']
#{'C': 3.2, 'gamma': 6e-06}
# {'C': 4.0, 'gamma': 1.2e-05} for 600pca

print('fourth')
#fourth
Cs = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
gammas = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
Cs = [x*C3 for x in Cs]
gammas = [x*gamma3 for x in gammas]
params4 = svc_param_selection2(X, Y, 5, Cs, gammas)
C4 = params3['C']
gamma4 = params3['gamma']
#{'C': 3.2, 'gamma': 6e-06}
#{'C': 3.2, 'gamma': 9.600000000000001e-06}#600pca

####################TO TEST
svm_object = svm.SVC(C=3.2, gamma=9.6e-06)
svm_object.fit(full_normPCA600_array[train_indicies],full_subClass_array[train_indicies])

svm_object.score(full_normPCA600_array[test_indicies],full_subClass_array[test_indicies])
predictions = svm_object.predict(full_normPCA600_array[test_indicies])

corr = is_correct(predictions,full_subClass_array[test_indicies])
N = np.size(full_subClass_array[test_indicies])
accuracy = np.sqrt(corr*(100-corr)/(100*N))



#################Creating the most beatiful graphs
pred = predictions
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

ax.set_xticklabels(x_ax,size = 20)
ax.set_yticklabels(y_ax,size = 20)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(y_ax)):
    for j in range(len(x_ax)):
        text = ax.text(j, i, str(np.round(confu_matrix_percent[i,j],1)) + " %", ha="center", va="center", color="black",size = 15)

ax.set_title("Predictet class compared to true class",size = 22)
fig.tight_layout()
plt.xlabel("True class",size = 22)
plt.ylabel("Predicted class",size = 22)
plt.show()




















#full_normalized_array


################PCA AND VARIANCE EXPLAINED
pca = PCA(svd_solver='auto')#PCA with all components
pca.fit(full_normalized_array[train_indicies])
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

###################1155 components for 99%
pca = PCA(svd_solver='auto', n_components = 1155)#PCA with all components
pca.fit(full_normalized_array[train_indicies])
full_normPCA1155_array = pca.transform(full_normalized_array)

#####################OPTIMIZE SVM
def svc_param_selection2(X, y, nfolds, Cs, gammas): #https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

Cs = [1e-2,1e-1,1e-0,1e+1,1e+2,1e+3,1e+4]
gammas = [1e-8,1e-7,1e-6, 1e-5,1e-4,1e-3,1e-2]
params = svc_param_selection2(full_normPCA1155_array[train_indicies], full_isAnimal_array[train_indicies], 5, Cs, gammas)
#{'C': 100.0, 'gamma': 1e-06}
#{'C': 10.0, 'gamma': 1e-05} for binary
Cs = [2,4,6,8,10,20,40,60,80]
gammas = [2e-6,4e-6,6e-6,8e-6,1e-5,2e-5,4e-5,6e-5,8e-5]
params = svc_param_selection2(full_normPCA1155_array[train_indicies], full_isAnimal_array[train_indicies], 5, Cs, gammas)
#{'C': 40, 'gamma': 4e-06}
#{'C': 4, 'gamma': 6e-06}
####################TO TEST
svm_object = svm.SVC(C=4, gamma=6e-6)
svm_object.fit(full_normPCA1155_array[train_indicies],full_isAnimal_array[train_indicies])

svm_object.score(full_normPCA1155_array[test_indicies],full_isAnimal_array[test_indicies])
predictions = svm_object.predict(full_normPCA1155_array[test_indicies])

corr = is_correct(predictions,full_isAnimal_array[test_indicies])
N = np.size(full_isAnimal_array[test_indicies])
accuracy = np.sqrt(corr*(100-corr)/(100*N))
#BINGO MAX


#################Creating the most beatiful graphs
pred = predictions
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

ax.set_title("Predictet class compared to true class of test set using SVM")
fig.tight_layout()
plt.xlabel("True class")
plt.ylabel("Predicted class")
plt.show()

