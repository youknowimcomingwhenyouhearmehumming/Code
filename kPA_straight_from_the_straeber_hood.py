# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:15:40 2019

@author: nikos
"""

#kPA
#######
#define nubmer of permutations, in the paper is 49
#!!!!!!!!!!!!!!!!Name you Training data:X_traintot!!!!!!!!!!!!!!!!!!!!!!!
Perm=49
#define H matrix
N=np.size(X_traintot,0)
one=np.ones([N,1])
H=np.eye(N)-(1/N)*np.dot(one,one.T)
#start permutations
sigma=10** np.linspace(-10,-1,10)

q=np.zeros(len(sigma),dtype='int32')
E=np.zeros(len(sigma))
for s in range(len(sigma)):
    #create list to keep the permutated arrays inside
   # K_perm_cen=[]
    #create matrix to put the eigenvalues of the permutated matrices
    lambdas=np.zeros([N,Perm])
    for p in range(Perm):
        X_perm=np.random.permutation(X_traintot)
        K_perm=sk.metrics.pairwise.rbf_kernel(X_perm,gamma=sigma[s])
        K_perm_cen=[H@K_perm@H]
        evl,egv=np.linalg.eig(K_perm_cen)
        evl_sort=np.sort(evl.real)
        evl_sort=evl_sort[::-1]
        lambdas[:,p]=evl_sort
    #compute original array kernel
    
    K_orig=sk.metrics.pairwise.rbf_kernel(X_traintot,gamma=sigma[s])
    K_or_cen=H@K_orig@H
    evl1,egv1=np.linalg.eig(K_perm_cen)
    evl1=evl1.reshape(N)
    evl_sort1=np.sort(evl1.real)
    evl_sort1=evl_sort1[::-1]

#compute and centerize kernel matrix step
    control=0
    indx=0
    T=np.zeros(N)
    while control==0:
        T[indx]=np.percentile(lambdas[indx,:],95)
        if (evl_sort1[indx]-T[indx])<0:
            q[s]=indx
            control=1
        indx=indx+1
    #calculate energy
    E[s]=np.sum(evl_sort1[0:q[s]]-T[0:q[s]])  
max_pos=np.argmax(E)
optimum_variance=sigma[max_pos] 
optimum_components=q[max_pos] 