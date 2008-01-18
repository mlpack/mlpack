Multiclass SVM Classifier

Examples:

1.cross validation mode
svm_bin --mode=cv --k_cv=4 --train_data=traindata.csv --kernel=gaussian --sigma=0.1 --c=1 --normalized=0
svm_bin --mode=cv --k_cv=4 --train_data=traindata.csv --kernel=linear --c=1 --normalized=0

2.training mode (model will be saved as "svm_model")
svm_bin --mode=train --train_data=traindata.csv --kernel=gaussian --sigma=0.1 --c=1 --normalized=0
svm_bin --mode=train --train_data=traindata.csv --kernel=linear --c=1 --normalized=0

3.training+testing mode
svm_bin --mode=train_test --train_data=traindata.csv --test_data=testdata.csv --kernel=gaussian --sigma=0.1 --c=1 --normalized=0
svm_bin --mode=train_test --train_data=traindata.csv --test_data=testdata.csv --kernel=linear --c=1 --normalized=0

4.testing mode (the model file "svm_model" shoud exist)
svm_bin --mode=test --test_data=testdata.csv --kernel=gaussian --sigma=0.1 --c=1 --normalized=0
svm_bin --mode=test --test_data=testdata.csv --kernel=linear --c=1 --normalized=0
