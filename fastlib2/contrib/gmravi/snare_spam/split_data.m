function[]=split_data()
data=csvread('packet_features_scaled.data');

[num_points,num_cols]=size(data);
%This splits the data into 80-20 and strips the train data of its labels
num_test_points=ceil(0.8*num_points);

p=randperm(num_points);

perm_matrix=sparse(zeros(num_points,num_cols));
for i=1:length(p);
    perm_matrix(i,p(i))=1;
end
data=perm_matrix*data;
%display(data);
%Write the train set
csvwrite('packet_features_scaled_train.data',data(1:num_test_points,1:end));

%Write the test set%
csvwrite('packet_features_scaled_test.data',data(num_test_points+1:end,1:end));

%Write the actual labels%
%csvwrite('packet_features_scaled_test_label.data',data(num_test_points+1:end,end:end));