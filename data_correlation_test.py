# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:20:06 2019

@author: Ralle
"""



import scipy.io
import numpy as np
import os
import matplotlib.pyplot as plt

def mean_all_class(data, class_vector):
    n_images, n_samples  = np.shape(data)
    class_mean_data = np.zeros((1,n_samples))
    
    classes = []
    for i in range(n_images):
        if not(class_vector[i] in classes):
            classes.append(str(class_vector[i]))
    
    n_classes = np.size(classes)
    classes_mean = np.zeros((n_classes,n_samples))
    for i in range(n_classes): #
        n_img_in_class = 0
        for j in range(n_images):
            if class_vector[j] == classes[i]:
                classes_mean[i] += data[j]
                n_img_in_class += 1
        classes_mean[i] = classes_mean[i]/n_img_in_class

    return classes_mean, classes

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

def corr_coeff_image(data):
    n_classes = np.size(data[0,:])
    for i in range(n_classes):
        for j in range(n_classes):
            
            


os.chdir('C:/Users/Ralle/Desktop/Advanced Machine Learning Project/AML/Nicolai/data')#


full_data_matrix = []
full_class_array = []
for i in range(nsubjects):
    eeg_events = scipy.io.loadmat('exp' + str(i+1) + '\eeg_events.mat')
    image_order = np.genfromtxt('exp' + str(i+1) + '\image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#
    data = eeg_events["eeg_events"]
    concat_data = concat_channels(data)
    if i == 0:
        full_data_matrix = concat_data
        full_class_array = image_order[:,1]
    else:
        full_data_matrix  = np.concatenate((full_data_matrix,concat_data))
        full_class_array  = np.concatenate((full_class_array,image_order[:,1]))


mean_data, classes = mean_all_class(full_data_matrix, full_class_array)

image = np.corrcoef(mean_data, mean_data,rowvar = 1)
plt.figure()
plt.xticks(range(23),classes,rotation = 60)
plt.yticks(range(23),classes,rotation = 60)
plt.imshow(image[0:23,0:23])
plt.title('Correlation between mean EEG of classes')
plt.show()

plt.figure()
plt.subplot(2,1,1)
plt.plot(mean_data[11,:])
plt.plot(mean_data[3,:])
plt.subplot(2,1,2)
plt.plot(mean_data[8,:])
plt.plot(mean_data[4,:])
plt.show()


##########################THE SAME but for ONE PERSON only
person_number  = 1 # 1 to 15
for i in range(15):
    person_number = i
    mean_data, classes = mean_all_class(full_data_matrix[(0+person_number*690):(690+person_number*690)], full_class_array[0:690])

    image = np.corrcoef(mean_data, mean_data,rowvar = 1)
    plt.figure()
    plt.xticks(range(23),classes,rotation = 90)
    plt.yticks(range(23),classes,rotation = 0)
    plt.imshow(image[0:23,0:23])
    plt.title('Correlation between mean EEG of classes for one person')
    plt.show()