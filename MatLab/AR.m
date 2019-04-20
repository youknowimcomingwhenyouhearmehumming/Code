clear
close all

chdir('C:/Users/Ralle/Desktop/Advanced Machine Learning Project/AML/Nicolai/data');
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

max_n_coeff = 50;
AIC_matrix = zeros(length(full_data(:,1)),max_n_coeff);

for j = 1:max_n_coeff
    for i = 1:length(full_data(:,1))
        th = ar(full_data(i,:),j,'burg');
        AIC_matrix(i,j) = aic(th);
    end
    j
end
%th = ar(new_data(1,:),n_coeffs)
%getcov(th)
%polydata(th)
%getpvec(th)%???
%b = aic(th)
%%
mean_AIC = mean(AIC_matrix,1);
figure()
plot(mean_AIC,'linewidth',3)
xlabel('number of coefficients')
title('mean AIC value')
