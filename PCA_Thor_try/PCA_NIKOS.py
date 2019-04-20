# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 20:37:43 2019

@author: nikos
"""
import numpy as np
import scipy.io as sio
import sklearn as sk
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
plt.close('all') 

os.chdir('C:/Users/Bruger/Documents/Uni/Advanche machine learning/Projekt/data_nikolai/Nicolai/data/exp1')

mat_data_subject1=sio.loadmat('eeg_events.mat')
mat_data_semantics1=sio.loadmat('image_semantics.mat')
mat_data_imorder1=sio.loadmat('image_order.mat')


#import the matrix of eeg events 32x550x690 of one specific subject
data_1=mat_data_subject1['eeg_events']
#plt.plot(data_1[:,:,1])
#plt.show()
#data1_2D=data_1.reshape(-1,data_1.shape[2])
#plt.plot(data1_2D[:,0])
#print(data_1[1,:,0]-data1_2D[550:1100,0])
#print(data1_2D[0:550,0])

pca=PCA(n_components=100)
P=pca.fit_transform(data1_2D)
print(np.sum(pca.explained_variance_ratio_))

#plt.scatter(P[:,0],P[:,1],s=1**2,c='red')
#plt.xlabel("PC1")  
#plt.ylabel("PC2")
#plt.title("PC1 vs PC2")



"""
-------------the first 100 and then all 690 eigenvectors are saved
"""

os.chdir('C:/Users/Bruger/Documents/Uni/Advanche machine learning/Projekt\Code\PCA_Thor_try')

pca.components_.dump("first_100_eigenvectors.dat") #Now the file .dat is saved to the open folder
#loaded_array = numpy.load("my_matrix.dat") #This import the file again

pca=PCA(n_components=690)
P=pca.fit_transform(data1_2D)
pca.components_.dump("all_690_eigenvectors.dat") #Now the file .dat is saved to the open folder




"""
-------------Plot of first 100 eigenvectors
"""
qutient_100=np.zeros(100)
for i in range(100):
    print(i)
    qutient_100[i]=sum(pca.explained_variance_ratio_[0:i])*100  
print (qutient_100[-1])
fig, ax1=plt.subplots(figsize=(9, 6)) 
ax1.plot(qutient_100,'.', color='b')
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.set_xlabel('Number of used principle components',fontsize=13)
ax1.set_ylabel('% of information$',fontsize=13)
ax1.set_title('Number of principle components. First 100 out of 550*32',fontsize=15)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.legend(loc='upper right')

