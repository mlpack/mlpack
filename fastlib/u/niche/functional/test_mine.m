% test script

c3 = 9;
c4 = 13;

mybasis = create_bspline_basis([-.63 -.13], 30, 4);

% y is num samples (500) by num epochs (318)
argvals = -.63:1/256:-.13;

lsize = size(left_epochs_data,3);

epochs_data = zeros(size(left_epochs_data) + [0 0 size(right_epochs_data, 3)]);


for i=1:lsize
  epochs_data(:,:,i) = left_epochs_data(:,:,i);
end

for i=1:size(right_epochs_data,3)
  epochs_data(:,:,i + lsize) = right_epochs_data(:,:,i);
end


myfd_train = data2fd(squeeze(epochs_data(13,:,:)), argvals, mybasis);

pca_results = pca_fd(myfd_train, 30);