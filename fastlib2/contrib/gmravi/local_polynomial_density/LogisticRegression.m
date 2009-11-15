function [a,b]=LR(train,test)
%So lets perform logistic regression with this data

%First we need to filter this data in order to make it suitable for
%experiments

[n_rows,n_cols]=size(train);

%Remove the last column as it has classification value

%This are the labels for the training data
true_classification_trg=train(1:n_rows, n_cols:n_cols);

vector_ones=ones(n_rows,1);

% To make training labels a number between 1 and k
true_classification_trg=true_classification_trg+vector_ones;

%This is the training data
trg_data=train(1:n_rows, 1:n_cols-1);

[number_of_trg_points,number_of_dimensions]=size(trg_data);

display('size of trg set is');
display(number_of_dimensions);

%%%%%%%Training data read complete%%%%%%%%%%%

%%%%%%%Read test data %%%%%%%%%%%%%%%%%%%%%%%%

[n_rows,n_cols]=size(test);

%Remove the last column as it has classification value
true_classification_test=test(1:n_rows, n_cols:n_cols);
test_data=test(1:n_rows, 1:n_cols-1);
[number_of_test_points,number_of_dimensions]=size(test_dat LogisticRegression(train,test)a);

%%%%%%%%Test data read %%%%%%%%%%%%%%%%%%%%%%%

%Fit a logistic regression model to the trg set. However note that Logistic
%regression requires class labels to be in 1-to-k. Whereas we use 0-1
%notation. hence add 1 to all the rows of the vector
%true_classification_trg

weights=mnrfit(train,true_classification_trg);
a=1;b=2;


