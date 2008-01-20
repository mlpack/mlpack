Multiclass SVM Classifier

Examples:

1.cross validation mode
svm_main --mode=cv --k_cv=4 --train_data=traindata.csv --kernel=gaussian --sigma=0.1 --c=1 --normalize=1
svm_main --mode=cv --k_cv=4 --train_data=traindata.csv --kernel=linear --c=1 --normalize=1

2.training mode (model will be saved as "svm_model")
svm_main --mode=train --train_data=traindata.csv --kernel=gaussian --sigma=0.1 --c=1 --normalize=1
svm_main --mode=train --train_data=traindata.csv --kernel=linear --c=1 --normalize=1

3.training+testing mode
svm_main --mode=train_test --train_data=traindata.csv --test_data=testdata.csv --kernel=gaussian --sigma=0.1 --c=1 --normalize=1
svm_main --mode=train_test --train_data=traindata.csv --test_data=testdata.csv --kernel=linear --c=1 --normalize=1

4.testing mode (the model file "svm_model" shoud exist)
svm_main --mode=test --test_data=testdata.csv --kernel=gaussian --sigma=0.1 --c=1 --normalize=1
svm_main --mode=test --test_data=testdata.csv --kernel=linear --c=1 --normalize=1
