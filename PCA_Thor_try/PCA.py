from __future__ import division
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
plt.close('all') 

os.chdir('C:\Users\Bruger\Documents\Uni\Advanche machine learning\Projekt\data_nikolai\Nicolai\data\exp1')

mat_contents = sio.loadmat('eeg_events.mat')
data=mat_contents['eeg_events']
reshaped_data=data.reshape(32*550,690)
reshaped_data=np.transpose(reshaped_data)

"""
--------------------------Task 1   PCA -----------------------------------------------------------------
"""

x=reshaped_data
x_mean=np.mean(x,axis=0) #is a 1x13 matrix

S=np.dot(np.transpose(x-x_mean),(x-x_mean))*1/len(x)
(eigensvalues, eigenvectors)=np.linalg.eig(S)

index= np.argsort(eigensvalues)[::-1]
eigenvectors_sorted=eigenvectors[:,index] #It is writting in this way becaue otherweise matrix with eigienvektors are
#sorted by the row and not the colum. We want to sort by colum since that is the way our eigenvecotes are stored.

U=eigenvectors_sorted
#Matrice U i vores formeler er altså bare eigenvektorer matricen - dog sorteret efter deres eigenværdier,
#Forklaring på Uk: k er antallet af eigenvektorer vi bruger. 

k=32*550# is the total number of paramters
z=np.dot(x-x_mean,U[:,0:k])

#the following  are the spectrum plot
eigensvalues_sorted=eigensvalues[index]

fig, ax1=plt.subplots(figsize=(9, 6)) 
ax1.plot(eigensvalues_sorted,'.', color='b', label='Eigenvalues')
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.set_xlabel('i',fontsize=13)
ax1.set_ylabel('$ \lambda_i $',fontsize=13)
ax1.set_title('Eigenspectrum',fontsize=15)
ax1.tick_params(axis='both', which='major', labelsize=12)

ax1.legend(loc='upper right')




#max_k=10
#qutient_2=sum(eigensvalues_sorted[0:max_k])/sum(eigensvalues_sorted[0:k])
#print qutient_2


qutient_100=np.zeros(100)
for i in range(100):
    print(i)
    qutient_100[i]=sum(eigensvalues_sorted[0:i])/sum(eigensvalues_sorted[0:k])*100  
print qutient_100[-1]
fig, ax1=plt.subplots(figsize=(9, 6)) 
ax1.plot(qutient_100,'.', color='b')
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.set_xlabel('Number of used principle components',fontsize=13)
ax1.set_ylabel('% of information$',fontsize=13)
ax1.set_title('Number of principle components. First 100 out of 550*32',fontsize=15)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.legend(loc='upper right')


qutient_500=np.zeros(500)
for i in range(500):
    qutient_500[i]=sum(eigensvalues_sorted[0:i])/sum(eigensvalues_sorted[0:k])*100  
print qutient_500[-1]
fig, ax1=plt.subplots(figsize=(9, 6)) 
ax1.plot(qutient_500,'.', color='b')
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.set_xlabel('Number of used principle components',fontsize=13)
ax1.set_ylabel('% of information$',fontsize=13)
ax1.set_title('Number of principle components. First 100 out of 550*32',fontsize=15)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.legend(loc='upper right')