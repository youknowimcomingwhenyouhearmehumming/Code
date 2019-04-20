import scipy.io
import numpy as np
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,precision_score

import matplotlib.pyplot as plt
from sklearn import preprocessing
import os

plt.close('all')
os.chdir('C:/Users/Bruger/Documents/Uni/Advanche machine learning/Projekt/data_nikolai/Nicolai/data/exp1')


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

def is_animal(class_vector):#creates vector of 1 if animal and 0 if not
    n = np.size(class_vector)
    bin_vector = np.zeros(n)
    for i in range(n):
        if class_vector[i] == 'animal':
            bin_vector[i] = 1
        else:
            bin_vector[i] = 0
    return bin_vector


def is_correct(pred,y_test):# returns number of correct predictions
    n = np.size(y_test)
    percent_correct = 0
    for i in range(n):
        if (y_test[i] == 1) and (pred[i] > 0.5):
            percent_correct += 1
        elif (y_test[i] == 0) and (pred[i] <= 0.5):
            percent_correct += 1
    percent_correct = percent_correct/n*100
    return percent_correct

def PCA_func(input_data,dims): #for 2d data with one observation per row. not really neccesary to have as function
    pca = PCA(n_components=dims, svd_solver='full')
    pca.fit(input_data)
    return pca

def do_KNN(X_train, X_test, y_train, y_test, n_neigh):
    neigh = KNeighborsClassifier(n_neighbors=n_neigh)
    neigh.fit(X_train,y_train)
    pred = neigh.predict(X_test)
    return is_correct(pred,y_test)


eeg_events = scipy.io.loadmat('eeg_events.mat')
image_order = np.genfromtxt('image_order.txt', delimiter="\t", skip_header = True, dtype=(str))#

nChannels,nSamples,nImg = np.shape(eeg_events["eeg_events"])
data = eeg_events["eeg_events"]
new_data = concat_channels(data)#merge channels
new_data_normal = preprocessing.scale(new_data)#normalize

X_train_index, X_test_index, y_train_string, y_test_string = train_test_split(list(range(nImg)),image_order[:,0],test_size=0.2)
X_train = data_transformed[X_train_index,:]
X_test = data_transformed[X_test_index,:]
y_train = is_animal(y_train_string)
y_test = is_animal(y_test_string)

 
runs=1
weighted_average=np.zeros(runs)
for i in range(runs):
    mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=200)
    
    mlp.fit(X_train,y_train)
    MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(13, 13, 13), learning_rate='constant',
           learning_rate_init=0.0001, max_iter=500, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=None,
           shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False)
    
    predictions = mlp.predict(X_test)
    
#    print(confusion_matrix(y_test,predictions))
#    print(classification_report(y_test,predictions))
    
    weighted_average[i]=(precision_score(y_test,predictions,average='weighted'))
    weighted_average=np.array(weighted_average)

#    print(confusion_matrix(y_test,predictions))
#    print(classification_report(y_test,predictions))
    

def plot(data):
    fig, ax1=plt.subplots() #
    xmin = np.min(data) 
    xmax = np.max(data)
    Nbins=len(data)
    histogram=ax1.hist(data, bins=Nbins,range=(xmin, xmax), label='titel') #The reason why it 
    ax1.set_title('titel',fontsize=15)
    ax1.legend(loc='upper right')
    plt.savefig('name.pdf')
    print(np.sum(data)/len(data))


print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
plot(weighted_average)

predictions = mlp.predict(X_train)
print(confusion_matrix(y_train,predictions))
print(classification_report(y_train,predictions))






#X=data_transformed
#_,_, y, _ = train_test_split(list(range(nImg)),image_order[:,0],test_size=0.0)
#
#from sklearn.model_selection import KFold
#kf = KFold(n_splits=10)
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(13,13,13), random_state=1)
#for train_indices, test_indices in kf.split(X):
#    clf.fit(X[train_indices], y[train_indices])
#    print(clf.score(X[test_indices], y[test_indices]))
#    
#    
#    



