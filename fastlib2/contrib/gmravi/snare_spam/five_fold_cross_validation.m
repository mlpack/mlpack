function[]=five_fold_cross_validation()
data=csvread('aggr_all_features_scaled.data'); %This is the entire dataset
[num_points,num_features]=size(data);
folds=5;
for i=1:folds
    test_set=data((i-1)*(num_points/folds)+1:i*num_points/folds,1:end);
    if(i==1)
        train_set=data(num_points/folds+1:end,1:end);
    else
        train_set1=data(1:(i-1)*num_points/folds,1:end);
        train_set2=data(i*num_points/folds+1:end,1:end);
        train_set=[train_set1;train_set2];
    end
    str_train='aggr_all_features_scaled_train';
    str_train=[str_train,int2str(i),'.data'];
    
    str_test='aggr_all_features_scaled_test'; 
    str_test=[str_test,int2str(i),'.data'];
    
    csvwrite(str_train,train_set);
    csvwrite(str_test,test_set);
end