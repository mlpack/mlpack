
template <typename TKernel> void  LocalKernelMachines <TKernel>:: 
GetTheFold_(Dataset &cv_train_data,Dataset &cv_test_data,
	    Vector &cv_train_labels, Vector &cv_test_labels,index_t fold_num)
{
  
  // The crossvalidation folds
  
  printf("dset is ...\n");
  dset_.matrix().PrintDebug();
  dset_.SplitTrainTest(k_folds_,fold_num,random_permutation_array_list_,
		       &cv_train_data,&cv_test_data);
  
  // In the stiched dataset, the last column was the label column. .
  // So now remove the label column.
  
  
  RemoveLastRowFromMatrixInit_(cv_train_data.matrix(),cv_train_labels);
  RemoveLastRowFromMatrixInit_(cv_test_data.matrix(),cv_test_labels);
 
  printf("cv train labels is....\n");
  cv_train_labels.PrintDebug();
  
  printf("cv test labels is ....\n");
  cv_test_labels.PrintDebug();
  
}

template <typename TKernel> void LocalKernelMachines<TKernel>::
CrossValidateOverAllThreeParameters_(){
  
  printf("Will crossvalidate over all three parameters...\n");


}

template <typename TKernel> void  LocalKernelMachines<TKernel>::
CrossValidateOverSVMKernelBandwidthAndSmoothingKernelBandwidth_(){
  
  printf("Will crossvalidate over svm and smoothing...\n");
  
}


template <typename TKernel> void  LocalKernelMachines<TKernel>::
CrossValidateOverSVMKernelBandwidthAndLambda_(){


  printf("Will crossvalidate over svm and lambda...\n");    

}

template <typename TKernel> LocalKernelMachines<TKernel>::
GenerateSmoothingBandwidthVectorForCV_(){



}

  
template <typename TKernel> void  LocalKernelMachines<TKernel>::
GenerateLambdaVectorForCV_(){

}

template <typename TKernel> void  LocalKernelMachines<TKernel>::
CrossValidateOverSmoothingKernelBandwidthAndLambda_(){
  
  printf("Will crossvalidate over smoothing and lambda...\n");
  
  // For cv puproses
  printf("We will do crossvalidation ...\n");
  printf("svm_kernel is %d...\n",svm_kernel_);
  
  
  Vector smoothing_bandwidth_vector;
  Vector lambda_vector;
  
  GenerateSmoothingBandwidthVectorForCV_();
  GenerateLambdaVectorForCV_();
  
  index_t smoothing_bandwidth_vector_length=
    smoothing_bandwidth_vector.length();
  
  index_t lambda_vector_length=
    lambda_vector.length();
  
  
  for(index_t i=0;i<smoothing_bandwidth_vector_length;i++){
    
    for(index_t j=0;j<lambda_vector_length;j++){
      
      for(index_t fold_num=0;fold_num<k_folds_;fold_num++){
	
	
	Dataset cv_train_data,cv_test_data;
	Vector cv_train_labels,cv_test_labels;
	
	// Will get the train and test folds 
	GetTheFold_(cv_train_data,cv_test_data,cv_train_labels, 
		    cv_test_labels,fold_num);
	
	// This routine will take the train fold and the test fold
	// and solve the local SVM problem
	
	RunLocalKernelMachines_(cv_train_data.matrix(),cv_test_data.matrix(),
				cv_train_labels);
      }
    }
  }
}


template <typename TKernel> void  LocalKernelMachines<TKernel>::CrossValidateOverSmoothingKernelBandwidth_(){

  printf("Will crossvalidate over smoothing...\n");
    

}
 
template <typename TKernel> void  LocalKernelMachines<TKernel>::CrossValidateOverSVMKernelBandwidth_(){
  
  printf("Will crossvalidate over similarity...\n");
    
   
}  
 
template <typename TKernel> void  LocalKernelMachines<TKernel>::CrossValidateOverLambda_(){
   
  printf("Will crossvalidate over lambda...\n");
    
   
}


 
template <typename TKernel> void LocalKernelMachines<TKernel>::PerformCrossvalidation_(){
  
  if(svm_kernel_==SVM_LINEAR_KERNEL){
      
    printf("linear kernel detected...\n");
      
    if(cv_smoothing_kernel_bandwidth_flag_==1&&cv_lambda_flag_==1){
	
      CrossValidateOverSmoothingKernelBandwidthAndLambda_();
	
    }
    else{
      if(cv_smoothing_kernel_bandwidth_flag_==1){
	  
	// Only need to crossvalidate over the bandwidth of
	// the smoothing kernel 
	  
	CrossValidateOverSmoothingKernelBandwidth_(); 
      }
      else{
	  
	if(cv_lambda_flag_==1){
	    
	  //Only need to crossvalidate over the bandwidth of the 
	  // smoothing kernel 
	    
	  CrossValidateOverLambda_(); 
	}
	else{

	  //No crossvalidation involved

	} 
      }
    }
  }
  else{
      
    if(cv_smoothing_kernel_bandwidth_flag_==1&&cv_lambda_flag_==1&&
       cv_svm_kernel_bandwidth_flag_==1){
	
      CrossValidateOverAllThreeParameters_();
    }
    else{
      if(cv_smoothing_kernel_bandwidth_flag_==1&&cv_lambda_flag_==1){
	  
	//Only need to crossvalidate over the bandwidth of the smoothing kernel 
	  
	CrossValidateOverSmoothingKernelBandwidthAndLambda_(); 
      }
      else{
	  
	if(cv_svm_kernel_bandwidth_flag_==1&&cv_lambda_flag_==1){
	    
	  //Only need to crossvalidate over the bandwidth of the smoothing kernel 
	    
	  CrossValidateOverSVMKernelBandwidthAndLambda_(); 
	}
	else{

	  if(cv_svm_kernel_bandwidth_flag_==1&&cv_smoothing_kernel_bandwidth_flag_==1){
	      
	    //Only need to crossvalidate over the bandwidth of the smoothing kernel 
	      
	    CrossValidateOverSVMKernelBandwidthAndSmoothingKernelBandwidth_(); 
	  }

	  else{
	    //Crossvalidation over just 1 single variable
	      
	    if(cv_smoothing_kernel_bandwidth_flag_==1){
		
	      CrossValidateOverSmoothingKernelBandwidth_();
	    }
	    else{

	      if(cv_lambda_flag_==1){

		CrossValidateOverLambda_();
	      }
	      else{

		if(cv_svm_kernel_bandwidth_flag_==1){
		    
		  CrossValidateOverSVMKernelBandwidth_();
		}
		else{

		  //No crossvalidation.

		}
		  
	      }
	    }
	  }
	}
      }
    }
  }
    
  // We now have the optimal parameters.
}
 
template <typename TKernel> void LocalKernelMachines<TKernel>::PrepareForCrossValidation_(){
  
  // The purpose of this function is simply to setup the stage for crossvalidation. 
  
  // If no crossvalidation is required then this code will not be invoked.,
  
  // In this case we are crossvalidating. hence lets stich the 
  // train data with the train labels.
  
  Matrix stiched_train_data;
  AppendVectorToMatrix(train_labels_vector_,train_data_,stiched_train_data);
  
  void *random_permutation;
  random_permutation=(void *)malloc(num_train_points_*sizeof(index_t));
  math::MakeRandomPermutation(num_train_points_,(index_t *) random_permutation);
  
  // Lets typecast random_permutation as ArrayList <index_t>
  
  random_permutation_array_list_.InitAlias((int *)random_permutation,num_train_points_);
  
  for(index_t i=0;i<num_train_points_;i++){
    
    printf("%d....",((index_t *)random_permutation)[i]);
  }
  
  for(index_t i=0;i<num_train_points_;i++){
    
    printf("%d....",random_permutation_array_list_[i]);
  }
  
  printf("Stiched train data is....\n");
  stiched_train_data.PrintDebug();
  
  // Now set the matrix part of dset to the train dataset
  
  dset_.matrix().Copy(stiched_train_data);
  
  printf("training data is...\n");
  dset_.matrix().PrintDebug();
}
   
template <typename TKernel> void LocalKernelMachines< TKernel>::SetUpCrossValidationFlags_(){
    
  // Check what all parameters we need to crossvalidate over.
    
  if (svm_kernel_==SVM_RBF_KERNEL){
      
    if(optimal_svm_kernel_bandwidth_<0){
	
      // Then we need to crossvalidate
	
      cv_svm_kernel_bandwidth_flag_=1;
	
      printf("Will crossvalidate for the bandwidth of RBF kernel...\n");
    }
    else{
	
      cv_svm_kernel_bandwidth_flag_=0;
	
      printf("NO need to  crossvalidate for the bandwidth of RBF kernel...\n");
    }
  }
  else{
      
    // SVM kernel is a linear kernel A linear kernel has no
    // bandwidth so we dont need to crossvalidate
      
    cv_svm_kernel_bandwidth_flag_=0;
    printf("No need to crossvalidate since this is a linear kernel...\n");
  }
    
  // Check if we need to crossvalidate for smoothing kernel 
  if(optimal_smoothing_kernel_bandwidth_<0){
      
    cv_smoothing_kernel_bandwidth_flag_=1;	
      
    printf("Will crossvalidate for the optimal bandwidth of the smoothing kernel...\n");
  }
  else{
      
    // This parameter is already provided. Hence do not
    // crossvalidate
    cv_smoothing_kernel_bandwidth_flag_=0;	
    printf("No need to crossvalidate for the optimal bandwidth of the smoothing kernel...\n");
  }
    
  // Check if we need to crossvalidate for lambda 
  if(optimal_lambda_<0){
      
    cv_lambda_flag_=1;	
    printf("Will have to crossvalidate for the optimal regularization value....\n");
  }
  else{
      
    // This parameter is already provided. Hence do not
    // crossvalidate
    cv_lambda_flag_=0;	
    printf("NO need to  crossvalidate for the optimal regularization value....\n");
      
  }
}





template <typename TKernel> void LocalKernelMachines<TKernel>::CrossValidation_(){
  
  SetUpCrossValidationFlags_();

  if(cv_svm_kernel_bandwidth_flag_!=0||
     cv_smoothing_kernel_bandwidth_flag_!=0||
     cv_lambda_flag_!=0 )
    {
      
      // This means there is atleast one parameter over 
      // which we need to crossvalidate 
      PrepareForCrossValidation_();
      PerformCrossValidation_();
    }
  else{
    
    // Straight away run the local kernel machine with the given parameters.
    
    // We don't use dset here. Hence set it to empty so as to avoid segfault
    
    dset_.InitBlank();
    
  }
  
}
