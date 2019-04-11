clear
close all

chdir('C:/Users/Ralle/Desktop/Advanced Machine Learning Project/AML/Nicolai/data');
datap1 = load('exp1/eeg_events');
data = datap1(1).eeg_events;
[n_channels, n_samples, n_images] = size(datap1(1).eeg_events);
new_data = zeros(n_images,n_channels*n_samples);
for j = 1:n_images
    for i = 1:n_channels
        new_data(j,1+(i-1)*n_samples:i*n_samples) = data(i,:,j);
    end
end


n_coeffs = 10
th = ar(new_data(1,:),n_coeffs)
getcov(th)
polydata(th)
getpvec(th)%???
aic(th)


