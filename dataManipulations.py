# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:55:36 2019

@author: Ralle
"""

import matplotlib.pyplot as plt
import scipy.io
import numpy as np


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

def mean_all_superclass(superclass_to_mean, nChannels, nSamples):#mean all observations with the given superclass . returns matrix channels*EEG_value*img = 32*550
    superclass_mean_data = np.zeros((nChannels,nSamples))
    n_images = 0
    for i in range(15): #number of test subjects
        eeg_events = scipy.io.loadmat('AML/Nicolai/data/exp' + str(i+1) + '/eeg_events.mat')
        image_order = np.genfromtxt('AML/Nicolai/data/exp' + str(i+1) + '/image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#
        data = eeg_events["eeg_events"]
        for j in range(np.size(image_order[:,0])): #for each image
            if image_order[j,0] == superclass_to_mean:#vector with the superclasses in the image order
                superclass_mean_data += data[:,:,j]
                n_images += 1
    superclass_mean_data = superclass_mean_data/n_images
    return superclass_mean_data

def mean_all_NOT_superclass(superclass_to_NOT_mean, nChannels, nSamples):#mean all observations with NOT the given superclass . returns matrix channels*EEG_value*img = 32*550
    superclass_mean_data = np.zeros((nChannels,nSamples))
    n_images = 0
    for i in range(15): #number of test subjects
        eeg_events = scipy.io.loadmat('AML/Nicolai/data/exp' + str(i+1) + '/eeg_events.mat')
        image_order = np.genfromtxt('AML/Nicolai/data/exp' + str(i+1) + '/image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#
        data = eeg_events["eeg_events"]
        for j in range(np.size(image_order[:,0])): #for each image
            if image_order[j,0] != superclass_to_NOT_mean:#vector with the superclasses in the image order
                superclass_mean_data += data[:,:,j]
                n_images += 1
    superclass_mean_data = superclass_mean_data/n_images
    return superclass_mean_data

def mean_channels_EEG(EEG_to_mean): #nchannels*nsamples
    nchannels = np.size(EEG_to_mean[:,0])
    mean_EEG = np.sum(EEG_to_mean,axis=0)/nchannels
    return mean_EEG
    

def plot_eeg_channels(data_to_plot,xlabel,title): #nchannels*nsamples
    plt.figure()
    nchanels = np.size(data_to_plot[:,0])
    for i in range(nchanels):
        plt.subplot(nchanels,1,i+1)
        plt.plot(data_to_plot[i,:])
        if i == 0:
            plt.title(title)
    plt.xlabel(xlabel)
    plt.show()
    
def plot_2_EEGs(eeg1,eeg2,xlabel,title,legend1,legend2):
    plt.figure()
    nchanels = np.size(eeg1[:,0])
    for i in range(nchanels):
        plt.subplot(nchanels,1,i+1)
        plt.plot(eeg1[i,:])
        plt.plot(eeg2[i,:])
        if i == 0:
            plt.title(title)
            plt.legend((legend1,legend2))
    plt.xlabel(xlabel)
    plt.show()

###############Load in the data
eeg_events = scipy.io.loadmat('AML/Nicolai/data/exp1/eeg_events.mat')
image_order = np.genfromtxt('AML/Nicolai/data/exp1/image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#
data = eeg_events["eeg_events"]
nChannels,nSamples,nImg = np.shape(eeg_events["eeg_events"])
    
mean_animal = mean_all_superclass('animal',32,550)
mean_NOT_animal = mean_all_NOT_superclass('animal',32,550)

plot_eeg_channels(mean_animal,'time ms', 'mean of animal EEG')
plot_eeg_channels(mean_NOT_animal,'time ms', 'mean of NOT animal EEG')

plot_2_EEGs(mean_animal,mean_NOT_animal,'time ms', 'mean of animal vs not animal EEG','animal','not animal')

mean_chan_animal = mean_channels_EEG(mean_animal)
mean_chan_NOT_animal = mean_channels_EEG(mean_NOT_animal)
plt.figure()
plt.plot(mean_chan_animal)
plt.plot(mean_chan_NOT_animal)
plt.show()






