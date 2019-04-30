

clear
close all

chdir('C:/Users/Ralle/Desktop/Advanced Machine Learning Project/AML/Nicolai/data');
addpath('C:\Users\Ralle\Desktop\Advanced Machine Learning Project\iqiukp-Relevance-Vector-Machine-c7724af')
datap1 = load('exp1/eeg_events');
data = datap1(1).eeg_events;
[n_channels, n_samples, n_images] = size(datap1(1).eeg_events);
n_subjects = 15;
full_data = zeros(n_images*n_subjects,n_channels*n_samples);
for p = 1:n_subjects
    %load a pesons data
    datap1 = load('exp' + string(p) + '/eeg_events');
    data = datap1(1).eeg_events;
    %concat the data into full data
    for j = 1:n_images
        for i = 1:n_channels
           full_data(j+(p-1)*n_images,1+(i-1)*n_samples:i*n_samples) = data(i,:,j);
        end
    end
end

%%
p1_data = full_data(1:690,:);


%% Fourier transform
Y = fft(p1_data);


figure()
plot(Y(1,:))

%%
ndims = 10;
[COEFF,SCORE] = pca(p1_data,'NumComponents',ndims);
pca_data = SCORE(:,1:ndims);


