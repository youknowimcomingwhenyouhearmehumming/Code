# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 20:37:43 2019

@author: nikos
"""
import mne
import numpy as np
import scipy.io as sio
import sklearn as sk
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
os.chdir('C:\Users\Bruger\Documents\Uni\Advanche machine learning\Projekt\data_nikolai\Nicolai\data\exp1')

mat_data_subject1=sio.loadmat('eeg_events.mat')
mat_data_semantics1=sio.loadmat('image_semantics.mat')
mat_data_imorder1=sio.loadmat('image_order.mat')


#import the matrix of eeg events 32x550x690 of one specific subject
data_1=mat_data_subject1['eeg_events']
plt.plot(data_1[:,:,1])
plt.show()
data1_2D=data_1.reshape(-1,data_1.shape[2])
plt.plot(data1_2D[:,0])
print(data_1[1,:,0]-data1_2D[550:1100,0])
print(data1_2D[0:550,0])
pca=PCA(n_components=100)
P=pca.fit_transform(data1_2D)
print(np.sum(pca.explained_variance_ratio_))

plt.scatter(P[:,0],P[:,1],s=1**2,c='red')
plt.xlabel("PC1")  
plt.ylabel("PC2")
plt.title("PC1 vs PC2")
