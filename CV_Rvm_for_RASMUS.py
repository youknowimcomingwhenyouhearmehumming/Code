# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 12:03:18 2019

@author: nikos
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:17:50 2019

@author: nikos
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 20:37:43 2019

@author: nikos
"""
import random
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


##################################################################################################################
def data_preparation():
#DATA PREPARATION AND INSERTATION
############################################################################################################################3
#FIRST SUBJECT
    mat_data_subject1=sio.loadmat('eeg_events1.mat')
    mat_data_semantics1=sio.loadmat('image_semantics.mat')
    mat_data_imorder1=sio.loadmat('image_order.mat')
#image order for classification
    data=pd.read_csv('image_order1.txt')
    new = data["supercategory	category	image_id"].str.split("	", n = 3, expand = True).astype(str)
    labels_An_tot=np.array(new).astype(str)
    labels_An=labels_An_tot[:,0]
    data_1=mat_data_subject1['eeg_events']
#NxD matrix, where N is the number of different variables(pictures=690) and D=17600 variable dimensionss
    data1_2D=data_1.reshape(-1,data_1.shape[2]).T #NxD matrix

#SECOND SUBJECT
    mat_data_subject2=sio.loadmat('eeg_events2.mat')
    data2=pd.read_csv('image_order2.txt')
    new2 = data2["supercategory	category	image_id"].str.split("	", n = 3, expand = True).astype(str)
    labels_An_tot2=np.array(new2).astype(str)
    labels_An2=labels_An_tot2[:,0]
    data_2=mat_data_subject2['eeg_events']
#NxD matrix, where N is the number of different variables(pictures=690) and D=17600 variable dimensionss
    data2_2D=data_2.reshape(-1,data_1.shape[2]).T #NxD matrix


#THIRD SUBJECT
    mat_data_subject3=sio.loadmat('eeg_events3.mat')
    data3=pd.read_csv('image_order3.txt')
    new3 = data3["supercategory	category	image_id"].str.split("	", n = 3, expand = True).astype(str)
    labels_An_tot3=np.array(new3).astype(str)
    labels_An3=labels_An_tot3[:,0]
    data_3=mat_data_subject3['eeg_events']
    #NxD matrix, where N is the number of different variables(pictures=690) and D=17600 variable dimensionss
    data3_2D=data_3.reshape(-1,data_1.shape[2]).T #NxD matrix

#FOURTH SUBJECT
    mat_data_subject4=sio.loadmat('eeg_events4.mat')
    data4=pd.read_csv('image_order4.txt')
    new4 = data4["supercategory	category	image_id"].str.split("	", n = 3, expand = True).astype(str)
    labels_An_tot4=np.array(new4).astype(str)
    labels_An4=labels_An_tot4[:,0]
    data_4=mat_data_subject4['eeg_events']
#NxD matrix, where N is the number of different variables(pictures=690) and D=17600 variable dimensionss
    data4_2D=data_4.reshape(-1,data_1.shape[2]).T #NxD matrix

#FIFTH SUBJECT
    mat_data_subject5=sio.loadmat('eeg_events5.mat')
    data5=pd.read_csv('image_order5.txt')
    new5 = data5["supercategory	category	image_id"].str.split("	", n = 3, expand = True).astype(str)
    labels_An_tot5=np.array(new5).astype(str)
    labels_An5=labels_An_tot5[:,0]
    data_5=mat_data_subject5['eeg_events']
#NxD matrix, where N is the number of different variables(pictures=690) and D=17600 variable dimensionss
    data5_2D=data_5.reshape(-1,data_1.shape[2]).T #NxD matrix



#6 SUBJECT
    mat_data_subject6=sio.loadmat('eeg_events6.mat')
    data6=pd.read_csv('image_order6.txt')
    new6 = data6["supercategory	category	image_id"].str.split("	", n = 3, expand = True).astype(str)
    labels_An_tot6=np.array(new6).astype(str)
    labels_An6=labels_An_tot6[:,0]
    data_6=mat_data_subject6['eeg_events']
#NxD matrix, where N is the number of different variables(pictures=690) and D=17600 variable dimensionss
    data6_2D=data_6.reshape(-1,data_1.shape[2]).T #NxD matrix

    #7 SUBJECT
    mat_data_subject7=sio.loadmat('eeg_events7.mat')
    data7=pd.read_csv('image_order7.txt')
    new7= data7["supercategory	category	image_id"].str.split("	", n = 3, expand = True).astype(str)
    labels_An_tot7=np.array(new7).astype(str)
    labels_An7=labels_An_tot7[:,0]
    data_7=mat_data_subject7['eeg_events']
    #NxD matrix, where N is the number of different variables(pictures=690) and D=17600 variable dimensionss
    data7_2D=data_7.reshape(-1,data_1.shape[2]).T #NxD matrix

#8 SUBJECT
    mat_data_subject8=sio.loadmat('eeg_events8.mat')
    data8=pd.read_csv('image_order8.txt')
    new8= data8["supercategory	category	image_id"].str.split("	", n = 3, expand = True).astype(str)
    labels_An_tot8=np.array(new8).astype(str)
    labels_An8=labels_An_tot8[:,0]
    data_8=mat_data_subject8['eeg_events']
    #NxD matrix, where N is the number of different variables(pictures=690) and D=17600 variable dimensionss
    data8_2D=data_8.reshape(-1,data_1.shape[2]).T #NxD matrix

#9 SUBJECT
    mat_data_subject9=sio.loadmat('eeg_events9.mat')
    data9=pd.read_csv('image_order3.txt')
    new9= data9["supercategory	category	image_id"].str.split("	", n = 3, expand = True).astype(str)
    labels_An_tot9=np.array(new9).astype(str)
    labels_An9=labels_An_tot9[:,0]
    data_9=mat_data_subject9['eeg_events']
#NxD matrix, where N is the number of different variables(pictures=690) and D=17600 variable dimensionss
    data9_2D=data_9.reshape(-1,data_1.shape[2]).T #NxD matrix

#10 SUBJECT
    mat_data_subject10=sio.loadmat('eeg_events10.mat')
    data10=pd.read_csv('image_order10.txt')
    new10= data10["supercategory	category	image_id"].str.split("	", n = 3, expand = True).astype(str)
    labels_An_tot10=np.array(new10).astype(str)
    labels_An10=labels_An_tot10[:,0]
    data_10=mat_data_subject10['eeg_events']
#NxD matrix, where N is the number of different variables(pictures=690) and D=17600 variable dimensionss
    data10_2D=data_10.reshape(-1,data_1.shape[2]).T #NxD matrix

#11 SUBJECT
    mat_data_subject11=sio.loadmat('eeg_events11.mat')
    data11=pd.read_csv('image_order11.txt')
    new11 = data11["supercategory	category	image_id"].str.split("	", n = 3, expand = True).astype(str)
    labels_An_tot11=np.array(new11).astype(str)
    labels_An11=labels_An_tot11[:,0]
    data_11=mat_data_subject11['eeg_events']
#NxD matrix, where N is the number of different variables(pictures=690) and D=17600 variable dimensionss
    data11_2D=data_11.reshape(-1,data_1.shape[2]).T #NxD matrix

#12 SUBJECT
    mat_data_subject12=sio.loadmat('eeg_events12.mat')
    data12=pd.read_csv('image_order12.txt')
    new12 = data12["supercategory	category	image_id"].str.split("	", n = 3, expand = True).astype(str)
    labels_An_tot12=np.array(new12).astype(str)
    labels_An12=labels_An_tot12[:,0]
    data_12=mat_data_subject12['eeg_events']
#NxD matrix, where N is the number of different variables(pictures=690) and D=17600 variable dimensionss
    data12_2D=data_12.reshape(-1,data_1.shape[2]).T #NxD matrix

#13 SUBJECT
    mat_data_subject13=sio.loadmat('eeg_events13.mat')
    data13=pd.read_csv('image_order13.txt')
    new13 = data13["supercategory	category	image_id"].str.split("	", n = 3, expand = True).astype(str)
    labels_An_tot13=np.array(new13).astype(str)
    labels_An13=labels_An_tot13[:,0]
    data_13=mat_data_subject13['eeg_events']
    #NxD matrix, where N is the number of different variables(pictures=690) and D=17600 variable dimensionss
    data13_2D=data_13.reshape(-1,data_1.shape[2]).T #NxD matrix

#14 SUBJECT
    mat_data_subject14=sio.loadmat('eeg_events14.mat')
    data14=pd.read_csv('image_order14.txt')
    new14 = data14["supercategory	category	image_id"].str.split("	", n = 3, expand = True).astype(str)
    labels_An_tot14=np.array(new14).astype(str)
    labels_An14=labels_An_tot14[:,0]
    data_14=mat_data_subject14['eeg_events']
#NxD matrix, where N is the number of different variables(pictures=690) and D=17600 variable dimensionss
    data14_2D=data_14.reshape(-1,data_1.shape[2]).T #NxD matrix

#15 SUBJECT
    mat_data_subject15=sio.loadmat('eeg_events15.mat')
    data15=pd.read_csv('image_order15.txt')
    new15= data15["supercategory	category	image_id"].str.split("	", n = 3, expand = True).astype(str)
    labels_An_tot15=np.array(new15).astype(str)
    labels_An15=labels_An_tot15[:,0]
    data_15=mat_data_subject15['eeg_events']
    #NxD matrix, where N is the number of different variables(pictures=690) and D=17600 variable dimensionss
    data15_2D=data_15.reshape(-1,data_1.shape[2]).T #NxD matrix


#merge LABELS
    tot=np.concatenate((labels_An,labels_An2,labels_An3,labels_An4,labels_An5,labels_An6,labels_An7,labels_An8,labels_An9,labels_An10,labels_An11,labels_An12,labels_An13,labels_An14,labels_An15),axis=0)

#MERGE DATA OF THE TEST SUBJECTS
    d_tot=np.concatenate((data1_2D,data2_2D,data3_2D,data4_2D,data5_2D,data6_2D,data7_2D,data8_2D,data9_2D,data10_2D,data11_2D,data12_2D,data13_2D,data14_2D,data15_2D),axis=0)
##########################################################################################################################################
###########################################################################################################################################################
    return (tot,d_tot)
#tot,d_tot=data_preparation()
    

#create function to get the transformed labels for classification, 1 for animal, 0 else
def Animal_label(labels):
    new=np.zeros(np.size(labels))
    for i in range(len(labels)):
        if labels[i]==("animal"):
            new[i]=1
        #else:
         #   new[i]=0
        elif labels[i]==("vehicle"):
            new[i]=2   
        elif labels[i]==("outdoor"):
            new[i]=3   
        elif labels[i]==("furniture"):
            new[i]=4  
        elif labels[i]==("indoor"):
            new[i]=5  
        elif labels[i]==("food"):
            new[i]=6      
    return new       
        


## RASMUS WAS HERE:
tot = full_class_array
d_tot = full_data_matrix



############PCA################################################################
pca=PCA(n_components=10)
P=pca.fit_transform(d_tot)
#a=pca.get_covariance().shape
#plt.figure(2)
#print(pca.explained_variance_ratio_)
#plt.figure(3)
#plt.scatter(P[:,0],P[:,1],c='red')
#plt.xlabel("PC1")  
#plt.ylabel("PC2")
#plt.title("PC1 vs PC2")
################################################################################
#KERNEL PCA
kpca = KernelPCA(kernel="rbf",n_components=3)
X_kpca = kpca.fit_transform(d_tot)
X_kpca.shape
#get the target values
An_lab=Animal_label(tot) 
An_lab.shape  
#shuffle data to take them in random order
#split the data to trainning and test set
X_train, X_test, y_train, y_test =train_test_split(P, An_lab, test_size=0.33, shuffle=True)
feature_scaler = StandardScaler()  
X_train = feature_scaler.fit_transform(X_train)  
X_test = feature_scaler.transform(X_test)  
####################################################################################################################
#CROSS VALIDATION SPLIT IN K-folds
######################################################################################################################
kf = KFold(n_splits=5)
kf.get_n_splits(X_train,X_test)    
print(kf)
#CREATE WIDTHS FOR GRID SEARCH
width1= np.linspace(-5,4,10)
width=10**width1
#create matrix to input the scores and widths of each iteration
score_width=np.ones([len(width),2])
##################
#FIRST for loop gives the values for different widths, SECOND for loop does Kfold_validation
for i in range(len(width)):
    score=0
    for train_index, test_index in kf.split(X_train):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train1, X_test1 = X_train[train_index], X_train[test_index]
        y_train1, y_test1 =y_train[train_index], y_train[test_index]
        clf1=RVC(kernel='rbf',coef1=width[i])
        clf1.fit(X_train1,y_train1)
        score=score+clf1.score(X_test1,y_test1) 
    score_width[i,0]=score
    score_width[i,1]=width[i]
#########################################################################################################
idx=np.argmax(score_width[:,0])
best_width=score_width[idx,1]
