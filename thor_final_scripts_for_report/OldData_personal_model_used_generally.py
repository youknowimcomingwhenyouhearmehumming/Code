import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,confusion_matrix,precision_score
from sklearn import preprocessing
import scipy.io as sio
from collections import Counter

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

#def Label(class_vector):#creates vector of 1 if animal and 0 if not
#    n = np.size(class_vector)
#    bin_vector = np.zeros(n)
#    for i in range(n):
#        if class_vector[i] == 'animal':
#            bin_vector[i] = 0
#        if class_vector[i] == 'food':
#            bin_vector[i] = 1
#        if class_vector[i] == 'furniture':
#            bin_vector[i] = 2
#        if class_vector[i] == 'indoor':
#            bin_vector[i] = 3
#        if class_vector[i] == 'outdoor':
#            bin_vector[i] = 4
#        if class_vector[i] == 'vehicle':
#            bin_vector[i] = 5
#    return bin_vector


print('-------------------------------------PERSON 1-------------------------------------------------------')

"""
-------------- Data is imported, concated and normalized--------------------
"""
os.chdir('C:/Users/Bruger/Documents/Uni/Advanche machine learning/Projekt/data_nikolai/Nicolai/data/exp1')
eeg_events = sio.loadmat('eeg_events.mat')
image_order = np.genfromtxt('image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#
mat_data_semantics1=sio.loadmat('image_semantics.mat')
nChannels,nSamples,nImg = np.shape(eeg_events["eeg_events"])
data = eeg_events["eeg_events"]

data_concat = concat_channels(data)#merge channels into 690x(550x32)
data_concat_normal = preprocessing.scale(data_concat)#normalize

"""
-------------- PCA is applied to the data-------------------------------------
"""
number_of_principle_components=400#690 is all of them
pca=PCA(n_components=number_of_principle_components)
pca_data=pca.fit_transform(data_concat_normal)
print(np.sum(pca.explained_variance_ratio_)) # if 690 is chosen then this should be 1.0



"""
-------------- X and y for train and test is made-------------------------------------
"""
X_train_index, X_test_index, y_train_string, y_test_string = train_test_split(list(range(nImg)),image_order[:,0],test_size=0.2)
X_train = data_concat[X_train_index,:]
X_test = data_concat[X_test_index,:]
y_train = Label(y_train_string)
y_test = Label(y_test_string)
#

"""
-------------- TPOT does is magic-------------------------------------
#"""
from tpot import TPOTClassifier
clf=TPOTClassifier(verbosity=2,n_jobs=1) 
clf.fit(X_train,y_train)

print(clf.score(X_test,y_test))
predictions = clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
score1=clf.score(X_test,y_test)
confusion_matrix1=confusion_matrix(y_test,predictions)

print('Order of which element that was the most of=',Counter(y_test_string).keys())
print('How many that were of each unique element',Counter(y_test_string).values())



print('-------------------------------------PERSON 2-------------------------------------------------------')
"""
-------------- Data is imported, concated and normalized--------------------
"""
os.chdir('C:/Users/Bruger/Documents/Uni/Advanche machine learning/Projekt/data_nikolai/Nicolai/data/exp2')
eeg_events = sio.loadmat('eeg_events.mat')
image_order = np.genfromtxt('image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#
mat_data_semantics1=sio.loadmat('image_semantics.mat')
nChannels,nSamples,nImg = np.shape(eeg_events["eeg_events"])
data = eeg_events["eeg_events"]

data_concat = concat_channels(data)#merge channels into 690x(550x32)
data_concat_normal = preprocessing.scale(data_concat)#normalize

"""
-------------- PCA is applied to the data-------------------------------------
"""
number_of_principle_components=690 #690 is all of them
pca=PCA(n_components=number_of_principle_components)
pca_data=pca.fit_transform(data_concat_normal)
print(np.sum(pca.explained_variance_ratio_)) # if 690 is chosen then this should be 1.0

"""
-------------- X and y for train and test is made-------------------------------------
"""
X_train_index, X_test_index, y_train_string, y_test_string = train_test_split(list(range(nImg)),image_order[:,0],test_size=0.2)
X_train = data_concat[X_train_index,:]
X_test = data_concat[X_test_index,:]
y_train = Label(y_train_string)
y_test = Label(y_test_string)


"""
-------------- Predictions-------------------------------------
#"""
print('clf.score=',clf.score(X_test,y_test))
predictions = clf.predict(X_test)
print('Confusion Matrix=')
print(confusion_matrix(y_test,predictions))
score2=clf.score(X_test,y_test)
confusion_matrix2=confusion_matrix(y_test,predictions)

print('Order of which element that was the most of=',Counter(y_test_string).keys())
print('How many that were of each unique element',Counter(y_test_string).values())






print('-------------------------------------PERSON 3-------------------------------------------------------')
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

"""
-------------- PCA is applied to the data-------------------------------------
"""
number_of_principle_components=690 #690 is all of them
pca=PCA(n_components=number_of_principle_components)
pca_data=pca.fit_transform(data_concat_normal)
print(np.sum(pca.explained_variance_ratio_)) # if 690 is chosen then this should be 1.0

"""
-------------- X and y for train and test is made-------------------------------------
"""
X_train_index, X_test_index, y_train_string, y_test_string = train_test_split(list(range(nImg)),image_order[:,0],test_size=0.2)
X_train = data_concat[X_train_index,:]
X_test = data_concat[X_test_index,:]
y_train = Label(y_train_string)
y_test = Label(y_test_string)


"""
-------------- Predictions-------------------------------------
#"""
print('clf.score=',clf.score(X_test,y_test))
predictions = clf.predict(X_test)
print('Confusion Matrix=')
print(confusion_matrix(y_test,predictions))
score3=clf.score(X_test,y_test)
confusion_matrix3=confusion_matrix(y_test,predictions)

print('Order of which element that was the most of=',Counter(y_test_string).keys())
print('How many that were of each unique element',Counter(y_test_string).values())






print('-------------------------------------PERSON 4-------------------------------------------------------')
"""
-------------- Data is imported, concated and normalized--------------------
"""
os.chdir('C:/Users/Bruger/Documents/Uni/Advanche machine learning/Projekt/data_nikolai/Nicolai/data/exp4')
eeg_events = sio.loadmat('eeg_events.mat')
image_order = np.genfromtxt('image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#
mat_data_semantics1=sio.loadmat('image_semantics.mat')
nChannels,nSamples,nImg = np.shape(eeg_events["eeg_events"])
data = eeg_events["eeg_events"]

data_concat = concat_channels(data)#merge channels into 690x(550x32)
data_concat_normal = preprocessing.scale(data_concat)#normalize

"""
-------------- PCA is applied to the data-------------------------------------
"""
number_of_principle_components=690 #690 is all of them
pca=PCA(n_components=number_of_principle_components)
pca_data=pca.fit_transform(data_concat_normal)
print(np.sum(pca.explained_variance_ratio_)) # if 690 is chosen then this should be 1.0

"""
-------------- X and y for train and test is made-------------------------------------
"""
X_train_index, X_test_index, y_train_string, y_test_string = train_test_split(list(range(nImg)),image_order[:,0],test_size=0.2)
X_train = data_concat[X_train_index,:]
X_test = data_concat[X_test_index,:]
y_train = Label(y_train_string)
y_test = Label(y_test_string)


"""
-------------- Predictions-------------------------------------
#"""
print('clf.score=',clf.score(X_test,y_test))
predictions = clf.predict(X_test)
print('Confusion Matrix=')
print(confusion_matrix(y_test,predictions))
score4=clf.score(X_test,y_test)
confusion_matrix4=confusion_matrix(y_test,predictions)

print('Order of which element that was the most of=',Counter(y_test_string).keys())
print('How many that were of each unique element',Counter(y_test_string).values())







print('-------------------------------------PERSON 5-------------------------------------------------------')
"""
-------------- Data is imported, concated and normalized--------------------
"""
os.chdir('C:/Users/Bruger/Documents/Uni/Advanche machine learning/Projekt/data_nikolai/Nicolai/data/exp5')
eeg_events = sio.loadmat('eeg_events.mat')
image_order = np.genfromtxt('image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#
mat_data_semantics1=sio.loadmat('image_semantics.mat')
nChannels,nSamples,nImg = np.shape(eeg_events["eeg_events"])
data = eeg_events["eeg_events"]

data_concat = concat_channels(data)#merge channels into 690x(550x32)
data_concat_normal = preprocessing.scale(data_concat)#normalize

"""
-------------- PCA is applied to the data-------------------------------------
"""
number_of_principle_components=690 #690 is all of them
pca=PCA(n_components=number_of_principle_components)
pca_data=pca.fit_transform(data_concat_normal)
print(np.sum(pca.explained_variance_ratio_)) # if 690 is chosen then this should be 1.0

"""
-------------- X and y for train and test is made-------------------------------------
"""
X_train_index, X_test_index, y_train_string, y_test_string = train_test_split(list(range(nImg)),image_order[:,0],test_size=0.2)
X_train = data_concat[X_train_index,:]
X_test = data_concat[X_test_index,:]
y_train = Label(y_train_string)
y_test = Label(y_test_string)


"""
-------------- Predictions-------------------------------------
#"""
print('clf.score=',clf.score(X_test,y_test))
predictions = clf.predict(X_test)
print('Confusion Matrix=')
print(confusion_matrix(y_test,predictions))
score5=clf.score(X_test,y_test)
confusion_matrix5=confusion_matrix(y_test,predictions)

print('Order of which element that was the most of=',Counter(y_test_string).keys())
print('How many that were of each unique element',Counter(y_test_string).values())








print('-------------------------------------PERSON 6-------------------------------------------------------')
"""
-------------- Data is imported, concated and normalized--------------------
"""
os.chdir('C:/Users/Bruger/Documents/Uni/Advanche machine learning/Projekt/data_nikolai/Nicolai/data/exp6')
eeg_events = sio.loadmat('eeg_events.mat')
image_order = np.genfromtxt('image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#
mat_data_semantics1=sio.loadmat('image_semantics.mat')
nChannels,nSamples,nImg = np.shape(eeg_events["eeg_events"])
data = eeg_events["eeg_events"]

data_concat = concat_channels(data)#merge channels into 690x(550x32)
data_concat_normal = preprocessing.scale(data_concat)#normalize

"""
-------------- PCA is applied to the data-------------------------------------
"""
number_of_principle_components=690 #690 is all of them
pca=PCA(n_components=number_of_principle_components)
pca_data=pca.fit_transform(data_concat_normal)
print(np.sum(pca.explained_variance_ratio_)) # if 690 is chosen then this should be 1.0

"""
-------------- X and y for train and test is made-------------------------------------
"""
X_train_index, X_test_index, y_train_string, y_test_string = train_test_split(list(range(nImg)),image_order[:,0],test_size=0.2)
X_train = data_concat[X_train_index,:]
X_test = data_concat[X_test_index,:]
y_train = Label(y_train_string)
y_test = Label(y_test_string)


"""
-------------- Predictions-------------------------------------
#"""
print('clf.score=',clf.score(X_test,y_test))
predictions = clf.predict(X_test)
print('Confusion Matrix=')
print(confusion_matrix(y_test,predictions))
score6=clf.score(X_test,y_test)
confusion_matrix6=confusion_matrix(y_test,predictions)

print('Order of which element that was the most of=',Counter(y_test_string).keys())
print('How many that were of each unique element',Counter(y_test_string).values())






print('-------------------------------------PERSON 7-------------------------------------------------------')
"""
-------------- Data is imported, concated and normalized--------------------
"""
os.chdir('C:/Users/Bruger/Documents/Uni/Advanche machine learning/Projekt/data_nikolai/Nicolai/data/exp7')
eeg_events = sio.loadmat('eeg_events.mat')
image_order = np.genfromtxt('image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#
mat_data_semantics1=sio.loadmat('image_semantics.mat')
nChannels,nSamples,nImg = np.shape(eeg_events["eeg_events"])
data = eeg_events["eeg_events"]

data_concat = concat_channels(data)#merge channels into 690x(550x32)
data_concat_normal = preprocessing.scale(data_concat)#normalize

"""
-------------- PCA is applied to the data-------------------------------------
"""
number_of_principle_components=690 #690 is all of them
pca=PCA(n_components=number_of_principle_components)
pca_data=pca.fit_transform(data_concat_normal)
print(np.sum(pca.explained_variance_ratio_)) # if 690 is chosen then this should be 1.0

"""
-------------- X and y for train and test is made-------------------------------------
"""
X_train_index, X_test_index, y_train_string, y_test_string = train_test_split(list(range(nImg)),image_order[:,0],test_size=0.2)
X_train = data_concat[X_train_index,:]
X_test = data_concat[X_test_index,:]
y_train = Label(y_train_string)
y_test = Label(y_test_string)


"""
-------------- Predictions-------------------------------------
#"""
print('clf.score=',clf.score(X_test,y_test))
predictions = clf.predict(X_test)
print('Confusion Matrix=')
print(confusion_matrix(y_test,predictions))
score7=clf.score(X_test,y_test)
confusion_matrix7=confusion_matrix(y_test,predictions)

print('Order of which element that was the most of=',Counter(y_test_string).keys())
print('How many that were of each unique element',Counter(y_test_string).values())






print('-------------------------------------PERSON 8-------------------------------------------------------')
"""
-------------- Data is imported, concated and normalized--------------------
"""
os.chdir('C:/Users/Bruger/Documents/Uni/Advanche machine learning/Projekt/data_nikolai/Nicolai/data/exp8')
eeg_events = sio.loadmat('eeg_events.mat')
image_order = np.genfromtxt('image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#
mat_data_semantics1=sio.loadmat('image_semantics.mat')
nChannels,nSamples,nImg = np.shape(eeg_events["eeg_events"])
data = eeg_events["eeg_events"]

data_concat = concat_channels(data)#merge channels into 690x(550x32)
data_concat_normal = preprocessing.scale(data_concat)#normalize

"""
-------------- PCA is applied to the data-------------------------------------
"""
number_of_principle_components=690 #690 is all of them
pca=PCA(n_components=number_of_principle_components)
pca_data=pca.fit_transform(data_concat_normal)
print(np.sum(pca.explained_variance_ratio_)) # if 690 is chosen then this should be 1.0

"""
-------------- X and y for train and test is made-------------------------------------
"""
X_train_index, X_test_index, y_train_string, y_test_string = train_test_split(list(range(nImg)),image_order[:,0],test_size=0.2)
X_train = data_concat[X_train_index,:]
X_test = data_concat[X_test_index,:]
y_train = Label(y_train_string)
y_test = Label(y_test_string)


"""
-------------- Predictions-------------------------------------
#"""
print('clf.score=',clf.score(X_test,y_test))
predictions = clf.predict(X_test)
print('Confusion Matrix=')
print(confusion_matrix(y_test,predictions))
score8=clf.score(X_test,y_test)
confusion_matrix8=confusion_matrix(y_test,predictions)

print('Order of which element that was the most of=',Counter(y_test_string).keys())
print('How many that were of each unique element',Counter(y_test_string).values())








print('-------------------------------------PERSON 9-------------------------------------------------------')
"""
-------------- Data is imported, concated and normalized--------------------
"""
os.chdir('C:/Users/Bruger/Documents/Uni/Advanche machine learning/Projekt/data_nikolai/Nicolai/data/exp9')
eeg_events = sio.loadmat('eeg_events.mat')
image_order = np.genfromtxt('image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#
mat_data_semantics1=sio.loadmat('image_semantics.mat')
nChannels,nSamples,nImg = np.shape(eeg_events["eeg_events"])
data = eeg_events["eeg_events"]

data_concat = concat_channels(data)#merge channels into 690x(550x32)
data_concat_normal = preprocessing.scale(data_concat)#normalize

"""
-------------- PCA is applied to the data-------------------------------------
"""
number_of_principle_components=690 #690 is all of them
pca=PCA(n_components=number_of_principle_components)
pca_data=pca.fit_transform(data_concat_normal)
print(np.sum(pca.explained_variance_ratio_)) # if 690 is chosen then this should be 1.0

"""
-------------- X and y for train and test is made-------------------------------------
"""
X_train_index, X_test_index, y_train_string, y_test_string = train_test_split(list(range(nImg)),image_order[:,0],test_size=0.2)
X_train = data_concat[X_train_index,:]
X_test = data_concat[X_test_index,:]
y_train = Label(y_train_string)
y_test = Label(y_test_string)


"""
-------------- Predictions-------------------------------------
#"""
print('clf.score=',clf.score(X_test,y_test))
predictions = clf.predict(X_test)
print('Confusion Matrix=')
print(confusion_matrix(y_test,predictions))
score9=clf.score(X_test,y_test)
confusion_matrix9=confusion_matrix(y_test,predictions)

print('Order of which element that was the most of=',Counter(y_test_string).keys())
print('How many that were of each unique element',Counter(y_test_string).values())









print('-------------------------------------PERSON 10-------------------------------------------------------')
"""
-------------- Data is imported, concated and normalized--------------------
"""
os.chdir('C:/Users/Bruger/Documents/Uni/Advanche machine learning/Projekt/data_nikolai/Nicolai/data/exp10')
eeg_events = sio.loadmat('eeg_events.mat')
image_order = np.genfromtxt('image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#
mat_data_semantics1=sio.loadmat('image_semantics.mat')
nChannels,nSamples,nImg = np.shape(eeg_events["eeg_events"])
data = eeg_events["eeg_events"]

data_concat = concat_channels(data)#merge channels into 690x(550x32)
data_concat_normal = preprocessing.scale(data_concat)#normalize

"""
-------------- PCA is applied to the data-------------------------------------
"""
number_of_principle_components=690 #690 is all of them
pca=PCA(n_components=number_of_principle_components)
pca_data=pca.fit_transform(data_concat_normal)
print(np.sum(pca.explained_variance_ratio_)) # if 690 is chosen then this should be 1.0

"""
-------------- X and y for train and test is made-------------------------------------
"""
X_train_index, X_test_index, y_train_string, y_test_string = train_test_split(list(range(nImg)),image_order[:,0],test_size=0.2)
X_train = data_concat[X_train_index,:]
X_test = data_concat[X_test_index,:]
y_train = Label(y_train_string)
y_test = Label(y_test_string)


"""
-------------- Predictions-------------------------------------
#"""
print('clf.score=',clf.score(X_test,y_test))
predictions = clf.predict(X_test)
print('Confusion Matrix=')
print(confusion_matrix(y_test,predictions))
score10=clf.score(X_test,y_test)
confusion_matrix10=confusion_matrix(y_test,predictions)

print('Order of which element that was the most of=',Counter(y_test_string).keys())
print('How many that were of each unique element',Counter(y_test_string).values())









print('-------------------------------------PERSON 11-------------------------------------------------------')
"""
-------------- Data is imported, concated and normalized--------------------
"""
os.chdir('C:/Users/Bruger/Documents/Uni/Advanche machine learning/Projekt/data_nikolai/Nicolai/data/exp11')
eeg_events = sio.loadmat('eeg_events.mat')
image_order = np.genfromtxt('image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#
mat_data_semantics1=sio.loadmat('image_semantics.mat')
nChannels,nSamples,nImg = np.shape(eeg_events["eeg_events"])
data = eeg_events["eeg_events"]

data_concat = concat_channels(data)#merge channels into 690x(550x32)
data_concat_normal = preprocessing.scale(data_concat)#normalize

"""
-------------- PCA is applied to the data-------------------------------------
"""
number_of_principle_components=690 #690 is all of them
pca=PCA(n_components=number_of_principle_components)
pca_data=pca.fit_transform(data_concat_normal)
print(np.sum(pca.explained_variance_ratio_)) # if 690 is chosen then this should be 1.0

"""
-------------- X and y for train and test is made-------------------------------------
"""
X_train_index, X_test_index, y_train_string, y_test_string = train_test_split(list(range(nImg)),image_order[:,0],test_size=0.2)
X_train = data_concat[X_train_index,:]
X_test = data_concat[X_test_index,:]
y_train = Label(y_train_string)
y_test = Label(y_test_string)


"""
-------------- Predictions-------------------------------------
#"""
print('clf.score=',clf.score(X_test,y_test))
predictions = clf.predict(X_test)
print('Confusion Matrix=')
print(confusion_matrix(y_test,predictions))
score11=clf.score(X_test,y_test)
confusion_matrix11=confusion_matrix(y_test,predictions)

print('Order of which element that was the most of=',Counter(y_test_string).keys())
print('How many that were of each unique element',Counter(y_test_string).values())









print('-------------------------------------PERSON 12-------------------------------------------------------')
"""
-------------- Data is imported, concated and normalized--------------------
"""
os.chdir('C:/Users/Bruger/Documents/Uni/Advanche machine learning/Projekt/data_nikolai/Nicolai/data/exp12')
eeg_events = sio.loadmat('eeg_events.mat')
image_order = np.genfromtxt('image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#
mat_data_semantics1=sio.loadmat('image_semantics.mat')
nChannels,nSamples,nImg = np.shape(eeg_events["eeg_events"])
data = eeg_events["eeg_events"]

data_concat = concat_channels(data)#merge channels into 690x(550x32)
data_concat_normal = preprocessing.scale(data_concat)#normalize

"""
-------------- PCA is applied to the data-------------------------------------
"""
number_of_principle_components=690 #690 is all of them
pca=PCA(n_components=number_of_principle_components)
pca_data=pca.fit_transform(data_concat_normal)
print(np.sum(pca.explained_variance_ratio_)) # if 690 is chosen then this should be 1.0

"""
-------------- X and y for train and test is made-------------------------------------
"""
X_train_index, X_test_index, y_train_string, y_test_string = train_test_split(list(range(nImg)),image_order[:,0],test_size=0.2)
X_train = data_concat[X_train_index,:]
X_test = data_concat[X_test_index,:]
y_train = Label(y_train_string)
y_test = Label(y_test_string)


"""
-------------- Predictions-------------------------------------
#"""
print('clf.score=',clf.score(X_test,y_test))
predictions = clf.predict(X_test)
print('Confusion Matrix=')
print(confusion_matrix(y_test,predictions))
score12=clf.score(X_test,y_test)
confusion_matrix12=confusion_matrix(y_test,predictions)

print('Order of which element that was the most of=',Counter(y_test_string).keys())
print('How many that were of each unique element',Counter(y_test_string).values())








print('-------------------------------------PERSON 13-------------------------------------------------------')
"""
-------------- Data is imported, concated and normalized--------------------
"""
os.chdir('C:/Users/Bruger/Documents/Uni/Advanche machine learning/Projekt/data_nikolai/Nicolai/data/exp13')
eeg_events = sio.loadmat('eeg_events.mat')
image_order = np.genfromtxt('image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#
mat_data_semantics1=sio.loadmat('image_semantics.mat')
nChannels,nSamples,nImg = np.shape(eeg_events["eeg_events"])
data = eeg_events["eeg_events"]

data_concat = concat_channels(data)#merge channels into 690x(550x32)
data_concat_normal = preprocessing.scale(data_concat)#normalize

"""
-------------- PCA is applied to the data-------------------------------------
"""
number_of_principle_components=690 #690 is all of them
pca=PCA(n_components=number_of_principle_components)
pca_data=pca.fit_transform(data_concat_normal)
print(np.sum(pca.explained_variance_ratio_)) # if 690 is chosen then this should be 1.0

"""
-------------- X and y for train and test is made-------------------------------------
"""
X_train_index, X_test_index, y_train_string, y_test_string = train_test_split(list(range(nImg)),image_order[:,0],test_size=0.2)
X_train = data_concat[X_train_index,:]
X_test = data_concat[X_test_index,:]
y_train = Label(y_train_string)
y_test = Label(y_test_string)


"""
-------------- Predictions-------------------------------------
#"""
print('clf.score=',clf.score(X_test,y_test))
predictions = clf.predict(X_test)
print('Confusion Matrix=')
print(confusion_matrix(y_test,predictions))
score13=clf.score(X_test,y_test)
confusion_matrix13=confusion_matrix(y_test,predictions)

print('Order of which element that was the most of=',Counter(y_test_string).keys())
print('How many that were of each unique element',Counter(y_test_string).values())







print('-------------------------------------PERSON 14-------------------------------------------------------')
"""
-------------- Data is imported, concated and normalized--------------------
"""
os.chdir('C:/Users/Bruger/Documents/Uni/Advanche machine learning/Projekt/data_nikolai/Nicolai/data/exp14')
eeg_events = sio.loadmat('eeg_events.mat')
image_order = np.genfromtxt('image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#
mat_data_semantics1=sio.loadmat('image_semantics.mat')
nChannels,nSamples,nImg = np.shape(eeg_events["eeg_events"])
data = eeg_events["eeg_events"]

data_concat = concat_channels(data)#merge channels into 690x(550x32)
data_concat_normal = preprocessing.scale(data_concat)#normalize

"""
-------------- PCA is applied to the data-------------------------------------
"""
number_of_principle_components=690 #690 is all of them
pca=PCA(n_components=number_of_principle_components)
pca_data=pca.fit_transform(data_concat_normal)
print(np.sum(pca.explained_variance_ratio_)) # if 690 is chosen then this should be 1.0

"""
-------------- X and y for train and test is made-------------------------------------
"""
X_train_index, X_test_index, y_train_string, y_test_string = train_test_split(list(range(nImg)),image_order[:,0],test_size=0.2)
X_train = data_concat[X_train_index,:]
X_test = data_concat[X_test_index,:]
y_train = Label(y_train_string)
y_test = Label(y_test_string)


"""
-------------- Predictions-------------------------------------
#"""
print('clf.score=',clf.score(X_test,y_test))
predictions = clf.predict(X_test)
print('Confusion Matrix=')
print(confusion_matrix(y_test,predictions))
score14=clf.score(X_test,y_test)
confusion_matrix14=confusion_matrix(y_test,predictions)

print('Order of which element that was the most of=',Counter(y_test_string).keys())
print('How many that were of each unique element',Counter(y_test_string).values())








print('-------------------------------------PERSON 15-------------------------------------------------------')
"""
-------------- Data is imported, concated and normalized--------------------
"""
os.chdir('C:/Users/Bruger/Documents/Uni/Advanche machine learning/Projekt/data_nikolai/Nicolai/data/exp15')
eeg_events = sio.loadmat('eeg_events.mat')
image_order = np.genfromtxt('image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#
mat_data_semantics1=sio.loadmat('image_semantics.mat')
nChannels,nSamples,nImg = np.shape(eeg_events["eeg_events"])
data = eeg_events["eeg_events"]

data_concat = concat_channels(data)#merge channels into 690x(550x32)
data_concat_normal = preprocessing.scale(data_concat)#normalize

"""
-------------- PCA is applied to the data-------------------------------------
"""
number_of_principle_components=690 #690 is all of them
pca=PCA(n_components=number_of_principle_components)
pca_data=pca.fit_transform(data_concat_normal)
print(np.sum(pca.explained_variance_ratio_)) # if 690 is chosen then this should be 1.0

"""
-------------- X and y for train and test is made-------------------------------------
"""
X_train_index, X_test_index, y_train_string, y_test_string = train_test_split(list(range(nImg)),image_order[:,0],test_size=0.2)
X_train = data_concat[X_train_index,:]
X_test = data_concat[X_test_index,:]
y_train = Label(y_train_string)
y_test = Label(y_test_string)


"""
-------------- Predictions-------------------------------------
#"""
print('clf.score=',clf.score(X_test,y_test))
predictions = clf.predict(X_test)
print('Confusion Matrix=')
print(confusion_matrix(y_test,predictions))
score15=clf.score(X_test,y_test)
confusion_matrix15=confusion_matrix(y_test,predictions)

print('Order of which element that was the most of=',Counter(y_test_string).keys())
print('How many that were of each unique element',Counter(y_test_string).values())





#os.chdir('C:/Users/Bruger/Documents/Uni/Advanche machine learning/Projekt/Code/thor_final_scripts_for_report')
#clf.export('TPOT_export_autoML_newData_Tsfresh_6_label.py.py')
#error_rate=clf.score(X_test,y_test)
#number_observations=len(X_test)
#print('uncertanty=',np.sqrt((error_rate*(1-error_rate))/(number_observations)))
#def f(error_rate,number_observations):
#    return np.sqrt((error_rate*(1-error_rate))/(number_observations))