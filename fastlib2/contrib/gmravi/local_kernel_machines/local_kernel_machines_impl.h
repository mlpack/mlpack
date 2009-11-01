


void  LocalKernelMachines< typename TKernel>:: 
RunLocalKernelMachines_(Matrix &local_train_data, Matrix &local_test_data, 
			Vector &local_train_data_labels){
  
  
  
  
  
  
}


void LocalKernelMachines< typename TKernel>::TrainLocalKernelMachines(){

  Crossvalidation_();
  
  // Having performed crossvalidation (if it was required) 
  // we now have the necessary parameters

  RunLocalKernelMachines_();
}
  
void LocalKernelMachines< typename TKernel>::Init(Matrix &train_data,Matrix &test_data,Vector &train_labels_vector,
						  struct datanode *module_in){
    
  //Initialize
    
  module_=module_in;
  train_data_.Alias(train_data);
  test_data_.Alias(test_data);
    
  printf("Train labels vector in main function is...\n");
  train_labels_vector.PrintDebug();
  train_labels_vector_.Alias(train_labels_vector);
    
    
  num_train_points_=train_data_.n_cols();
  num_test_points_=test_data_.n_cols();
  num_dims_=train_data_.n_rows();

  //Just set the bandwidth to DBL_MIN

  optimal_smoothing_kernel_bandwidth_=
    fx_param_double(fx_root,"smoothing_kernel_bandwidth",-DBL_MAX);

    
  optimal_svm_kernel_bandwidth_=
    fx_param_double(fx_root,"svm_kernel_bandwidth",-DBL_MAX);


    
  //Get the regularization parameter

  optimal_lambda_= fx_param_double(fx_root,"lambda",-DBL_MAX);

  // Get the number of folds of crossvalidation
    
  k_folds_=fx_param_int(fx_root,"k_folds",2);
    
    
  // Find out the kernels being used

  const char *smoothing_kernel=
    fx_param_str(fx_root,"smoothing_kernel","epan");
    
  const char *svm_kernel=
    fx_param_str(fx_root,"svm_kernel","linear");

  if(strcmp(smoothing_kernel,"epan")==0){
      
    smoothing_kernel_=EPAN_SMOOTHING_KERNEL;
      
  }
  else{
      
    smoothing_kernel_=GAUSSIAN_SMOOTHING_KERNEL;
  }

  if(strcmp(svm_kernel,"linear")==0){

    printf("SVM kernel is linaer....\n");
      
    svm_kernel_=SVM_LINEAR_KERNEL;
      
  }
  else{
    printf("kernel is RBF...\n");
    svm_kernel_=SVM_RBF_KERNEL;
  }
    
  printf("svm_kernel=%d..\n",svm_kernel_);
  printf("Basic initializations done...\n");
}
