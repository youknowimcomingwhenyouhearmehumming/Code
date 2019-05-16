# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:52:18 2019

@author: nikos
"""


import math
from numpy import linalg as LA
from statsmodels.tsa.ar_model import AR
import numpy as np
from sklearn import svm
import scipy.io as sio
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import sklearn as sk
from skrvm import RVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA,KernelPCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

##################################################################################################################

#DATA PREPARATION AND INSERTATION
############################################################################################################################3
#FIRST SUBJECT
mat_data_subject1=sio.loadmat('eeg_events1.mat')
#image order for classification
data=pd.read_csv('image_order1.txt')
new = data["supercategory	category	image_id"].str.split("	", n = 3, expand = True).astype(str)
labels_An_tot=np.array(new).astype(str)
labels_An=labels_An_tot[:,0]
labels_classes=labels_An_tot[:,1]
data_1=mat_data_subject1['eeg_events']
#NxD matrix, where N is the number of different variables(pictures=690) and D=17600 variable dimensionss
data1_2D=data_1.reshape(-1,data_1.shape[2]).T #NxD matrix

#SECOND SUBJECT
mat_data_subject2=sio.loadmat('eeg_events2.mat')
data2=pd.read_csv('image_order2.txt')
new2 = data2["supercategory	category	image_id"].str.split("	", n = 3, expand = True).astype(str)
labels_An_tot2=np.array(new2).astype(str)
labels_An2=labels_An_tot2[:,0]
labels_classes2=labels_An_tot2[:,1]
data_2=mat_data_subject2['eeg_events']
#NxD matrix, where N is the number of different variables(pictures=690) and D=17600 variable dimensionss
data2_2D=data_2.reshape(-1,data_1.shape[2]).T #NxD matrix


#THIRD SUBJECT
mat_data_subject3=sio.loadmat('eeg_events3.mat')
data3=pd.read_csv('image_order3.txt')
new3 = data3["supercategory	category	image_id"].str.split("	", n = 3, expand = True).astype(str)
labels_An_tot3=np.array(new3).astype(str)
labels_An3=labels_An_tot3[:,0]
labels_classes3=labels_An_tot3[:,1]
data_3=mat_data_subject3['eeg_events']
   #NxD matrix, where N is the number of different variables(pictures=690) and D=17600 variable dimensionss
data3_2D=data_3.reshape(-1,data_1.shape[2]).T #NxD matrix

#FOURTH SUBJECT
mat_data_subject4=sio.loadmat('eeg_events4.mat')
data4=pd.read_csv('image_order4.txt')
new4 = data4["supercategory	category	image_id"].str.split("	", n = 3, expand = True).astype(str)
labels_An_tot4=np.array(new4).astype(str)
labels_An4=labels_An_tot4[:,0]
labels_classes4=labels_An_tot4[:,1]
data_4=mat_data_subject4['eeg_events']
#NxD matrix, where N is the number of different variables(pictures=690) and D=17600 variable dimensionss
data4_2D=data_4.reshape(-1,data_1.shape[2]).T #NxD matrix



#merge LABELS
tot=np.concatenate((labels_An,labels_An2,labels_An3,labels_An4),axis=0)
tot_classes=np.concatenate((labels_classes,labels_classes2,labels_classes3,labels_classes4),axis=0)

    #tot=labels_An
#MERGE DATA OF THE TEST SUBJECTS
    #d_tot=data1_2D
d_tot=np.concatenate((data1_2D,data2_2D,data3_2D,data4_2D),axis=0)
##########################################################################################################################################
###########################################################################################################################################################
   
#create function to get the transformed labels for classification, 1 for animal, 0 else
def Animal_label(labels):
 
    new=np.zeros(np.size(labels))
    for i in range(len(labels)):
        if labels[i]==("vehicle"):
            new[i]=0
#        else:
#            new[i]=1
        elif labels[i]==("animal"):
            new[i]=1   
        elif labels[i]==("food"):
            new[i]=2     
    return new   
def Animal_label2(labels):
    sh=0
    tr=0
    air=0
    ze=0
    sh=0
    ele=0
    pi=0
    new=np.zeros(np.size(labels))
    for i in range(len(labels)):
        if labels[i]==("train"):
            new[i]=0
            tr=tr+1
        elif labels[i]==("airplane"):
            new[i]=1   
            air=air+1
        elif labels[i]==("zebra"):
            new[i]=2  
            ze=ze+1
        elif labels[i]==("sheep"):
            new[i]=4 
            sh=sh+1
        elif labels[i]==("elephant"):
            new[i]=5  
            ele=ele+1
        elif labels[i]==("pizza"):
            new[i]=6 
            pi=pi+1
    return new        
        

An_lab=Animal_label(tot) 
An_lab_classes=Animal_label2(tot_classes) 
dims_to_keep = []
chan_numbers = np.concatenate((np.add(np.arange(12),20),(np.add(np.arange(8),57))))
for i in range(np.size(chan_numbers)):
    dims_to_keep = np.concatenate((dims_to_keep,np.add((chan_numbers[i]-1)*576,np.arange(576))))


dims_to_keep = dims_to_keep.astype(int)
data_reduced = d_tot[:,dims_to_keep]
d_tot=data_reduced
X_traintot, X_testtot, y_train, y_test =train_test_split(d_tot, An_lab, test_size=0.20, shuffle=False)



#feature scalling before PCA
feature_scaler = StandardScaler()  
X_traintot= feature_scaler.fit_transform(X_traintot) 
X_testtot= feature_scaler.transform(X_testtot) 
#APPLY KPA
Perm=49
#define H matrix
N=np.size(X_traintot,0)
C=np.size(X_traintot,1)
one=np.ones([N,1])
H=np.eye(N)-(1/N)*np.dot(one,one.T)
#start permutations
#sigma=10** np.linspace(-7,-1,10)
sigma=np.array([0.000001,0.00001,0.0001,0.001])
q=np.zeros(len(sigma),dtype='int32')
E=np.zeros(len(sigma))
X_perm=np.zeros([N,C])
for s in range(len(sigma)):
    #create list to keep the permutated arrays inside
   # K_perm_cen=[]
    #create matrix to put the eigenvalues of the permutated matrices
    lambdas=np.zeros([N,Perm])
    for p in range(Perm):
        for c in range(C):
            indx_p=np.random.permutation(N)
            X_perm[:,c]=X_traintot[indx_p,c]
        K_perm=sk.metrics.pairwise.rbf_kernel(X_perm,X_perm,gamma=sigma[s])
        K_perm_cen=H@K_perm@H
        evl,egv=np.linalg.eig(K_perm_cen)
        evl=evl.reshape(N)
        evl_sort=np.sort(evl.real)
        evl_sort=evl_sort[::-1]
        lambdas[:,p]=evl_sort
    #compute original array kernel
    
    K_orig=sk.metrics.pairwise.rbf_kernel(X_traintot,X_traintot,gamma=sigma[s])
    K_or_cen=H@K_orig@H
    evl1,egv1=np.linalg.eig(K_or_cen)
    evl1=evl1.reshape(N)
    evl_sort1=np.sort(evl1.real)
    evl_sort1=evl_sort1[::-1]
#compute and centerize kernel matrix step
    T=np.zeros(N)
    for i in range(N):
        T[i]=np.percentile(lambdas[i,:],95)
    indx=0
    while evl_sort1[indx]-T[indx]>0 and indx<len(evl_sort1)-1:
        indx=indx+1 
    q[s]=indx    
    #calculate energy
    E[s]=np.sum(evl_sort1[0:q[s]]-T[0:q[s]]) 
    plt.plot(evl_sort1[0:300])
    plt.plot(T[0:300])
max_pos=np.argmax(E)
optimum_variance=sigma[max_pos] 
optimum_components=q[max_pos] 
print(optimum_variance)
print(optimum_components) 
plt.plot(evl_sort1[0:300])
plt.plot(T[0:300])
##Transform
kpca = KernelPCA(kernel="rbf",n_components=207,gamma=10**-5)
X_train=kpca.fit_transform(X_traintot)
X_test=kpca.transform(X_testtot)
#RVM
clf1=SVC(kernel='linear')
clf1.fit(X_train,y_train)
clf1.score(X_train,y_train)     
print(clf1.score(X_test,y_test))

#SVM
kf = KFold(n_splits=5)
kf.get_n_splits(X_train)    
print(kf)
#CREATE WIDTHS FOR GRID SEARCH
width=np.array( [0.1,1,10,100,1000])
#create matrix to input the scores and widths of each iteration
score_width=np.zeros([len(width)])
##################
#FIRST for loop gives the values for different widths, SECOND for loop does Kfold_validation
for i in range(len(width)):
    score=0
    for train_index, test_index in kf.split(X_train):
        X_train1, X_test1 = X_train[train_index], X_train[test_index]
        y_train1, y_test1 =y_train[train_index], y_train[test_index]
        clf1=SVC(kernel='linear',C=width[i])
        clf1.fit(X_train1,y_train1)
        score=score+clf1.score(X_test1,y_test1) 
    score_width[i]=score
print(score_width)


clf1=SVC(kernel='linear',C=10,probability=True)
clf1.fit(X_train,y_train)
clf1.score(X_train,y_train)     
print(clf1.score(X_test,y_test))





