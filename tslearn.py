import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,precision_score

from sklearn.decomposition import PCA
from sklearn import preprocessing
import scipy.io as sio
plt.close('all')

def concat_channels(eeg_events):#channels*EEG_value*img
    #for putting all channels in a single vector. return is n_img x 17600
    n_channels,n_samples,n_img = np.shape(eeg_events)
    concat_all = np.zeros((n_img,n_channels*n_samples))
    for i in range(n_img): #for each image
        concat_row = []
        for j in range(n_channels): #for each channel of channels
            #concat_data[i] = np.concatenate((concat_data[i],eeg_events[j,:,i]),axis=1)
            concat_row = np.concatenate((concat_row,eeg_events[j,:,i]))
        concat_all[i] = concat_row
    return concat_all #[n_img*17600]

        
        
#def Label(labels):
#    label_vector=np.zeros(np.size(labels))
#    for i in range(len(labels)):
#        if labels[i]==("animal"):
#            label_vector[i]=1
#        elif labels[i]==("vehicle"):
#            label_vector[i]=2   
#        elif labels[i]==("outdoor"):
#            label_vector[i]=3   
#        elif labels[i]==("furniture"):
#            label_vector[i]=4  
#        elif labels[i]==("indoor"):
#            label_vector[i]=5  
#        elif labels[i]==("food"):
#            label_vector[i]=6      
#    return label_vector      

def Label(class_vector):#creates vector of 1 if animal and 0 if not
    n = np.size(class_vector)
    bin_vector = np.zeros(n)
    for i in range(n):
        if class_vector[i] == 'animal':
            bin_vector[i] = 1
        else:
            bin_vector[i] = 0
    return bin_vector

"""
-------------- Data is imported, concated and normalized--------------------
"""
os.chdir('C:/Users/Bruger/Documents/Uni/Advanche machine learning/Projekt/data_nikolai/Nicolai/data/exp3')
eeg_events = sio.loadmat('eeg_events.mat')
image_order = np.genfromtxt('image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#
mat_data_semantics1=sio.loadmat('image_semantics.mat')
nChannels,nSamples,nImg = np.shape(eeg_events["eeg_events"])
data = eeg_events["eeg_events"]

data_concat = concat_channels(data)#merge channels into 690x(550x32)
data_concat_normal = preprocessing.scale(data_concat)#normalize
#
#"""
#-------------- PCA is applied to the data-------------------------------------
#"""
#number_of_principle_components=690 #690 is all of them
#pca=PCA(n_components=number_of_principle_components)
#pca_data=pca.fit_transform(data_concat_normal)
#print(np.sum(pca.explained_variance_ratio_)) # if 690 is chosen then this should be 1.0

"""
-------------- X and y for train and test is made-------------------------------------
"""
X_train_index, X_test_index, y_train_string, y_test_string = train_test_split(list(range(nImg)),image_order[:,0],test_size=0.2)
X_train = data_concat_normal[X_train_index,:]
X_test = data_concat_normal[X_test_index,:]
y_train = Label(y_train_string)
y_test = Label(y_test_string)

print(sum(y_test==1))
print(len(y_test))
print('percentage of not animals=',(sum(y_test==1)-len(y_test))/len(y_test))



"""
-------------- TSlearn------------------------------------
"""
from tslearn.utils import to_time_series
my_first_time_series = [1, 3, 4, 2]
formatted_time_series = to_time_series(my_first_time_series)
print(formatted_time_series.shape)

from tslearn.metrics import dtw
dtw(data_concat)
dtw(data_concat[0,:], data_concat[1,:])

#cdist_dtw(data_concat[0,:],[data_concat[1,:]])



"""
-------------- TPOT does is magic-------------------------------------
"""
#
#
#print(sum(y_test==1))
#print(len(y_test))
#print('percentage of not animals=',(sum(y_test==1)-len(y_test))/len(y_test))
#
#from tpot import TPOTClassifier
#clf=TPOTClassifier(verbosity=2,n_jobs=-1)
#clf.fit(X_train,y_train)
#
#print(sum(y_test==1))
#print(len(y_test))
#print('percentage of not animals=',(sum(y_test==1)-len(y_test))/len(y_test))
#
#
#print('test score=',clf.score(X_test,y_test))
#predictions = clf.predict(X_test)
#print(confusion_matrix(y_test,predictions))


#digits=load_digits()
#X=digits['data']
#y=digits['target']
#
#X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y)
#
#from sklearn.linear_model import LogisticRegression
#clf = LogisticRegression()
#clf.fit(X_train,y_train)
#
#result=clf.score(X_test,y_test)
#print (result)

#from tpot import TPOTClassifier
#clf=TPOTClassifier(verbosity=2,n_jobs=-1,generations=5,config_dict='TPOT light', max_time_mins=2)
#clf.fit(X_train,y_train)