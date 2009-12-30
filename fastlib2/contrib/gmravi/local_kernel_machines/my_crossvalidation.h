#include "utils.h"
#ifndef MY_CROSSVALIDATION_H
#define MY_CROSSVALIDATION_H


/////////////////// Set up the parameters for Crossvalidation//////////////////////

template <typename TKernel> void  LocalKernelMachines <TKernel>::
GenerateSmoothingBandwidthVectorForCV_(Vector &smoothing_bandwidth_vector){
  
  smoothing_bandwidth_vector.Init(1);
  for(index_t i=0; i<1;i++){
    smoothing_bandwidth_vector[i]=0.1*(1+i);
  }

}

template <typename TKernel> void  LocalKernelMachines <TKernel>::
GenerateLambdaVectorForCV_(Vector &lambda_vector){
  
 lambda_vector.Init(1);
  for(index_t i=0; i<1;i++){
    lambda_vector[i]=(1+i);
  }
 
}
template <typename TKernel> void  LocalKernelMachines <TKernel>::
GenerateSVMBandwidthVectorForCV_(Vector &svm_bandwidth_vector){
  
  svm_bandwidth_vector.Init(1);
  for(index_t i=0; i<1;i++){
    svm_bandwidth_vector[i]=0.4+0.1*i;
  }
  
}
/////////////////////////////////////////////////////////////////////////////////

 
template <typename TKernel> void  LocalKernelMachines <TKernel>:: 
GetTheFold_(Dataset &cv_train_data,Dataset &cv_test_data,
	    Vector &cv_train_labels, Vector &cv_test_labels,index_t fold_num)
{
  
  // The crossvalidation folds
  
  dset_.SplitTrainTest(k_folds_,fold_num,random_permutation_array_list_,
		       &cv_train_data,&cv_test_data);
  
  // In the stiched dataset, the last column was the label column. .
  // So now remove the label column.
  
  // printf("Removing last row from matrix....\n");
  RemoveLastRowFromMatrixInit(cv_train_data.matrix(),cv_train_labels);
  //printf("After removing last row we have.....\n");
  //printf("The train data is....\n");
  //  cv_train_data.matrix().PrintDebug();

  //printf("The labels are....\n");
  //cv_train_labels.PrintDebug();

  RemoveLastRowFromMatrixInit(cv_test_data.matrix(),cv_test_labels);
  
}

template <typename TKernel> void LocalKernelMachines<TKernel>::
CrossValidateOverSmoothingKernelBandwidthAndLambda_(){
  
  printf("Will crossvalidate over smoothing and lambda...\n");
  
  
  Vector smoothing_kernel_bandwidth_vector;
  Vector lambda_vector;
  
  GenerateSmoothingBandwidthVectorForCV_(smoothing_kernel_bandwidth_vector);
  GenerateLambdaVectorForCV_(lambda_vector);
  
  index_t smoothing_kernel_bandwidth_vector_length=
    smoothing_kernel_bandwidth_vector.length();
  
  index_t lambda_vector_length=
    lambda_vector.length();
  
  printf("Smoothing kernel bandwidth vector is...\n");
  smoothing_kernel_bandwidth_vector.PrintDebug();
  
  printf("Lambda vector is...\n");
  lambda_vector.PrintDebug();
  
  
  optimal_svm_kernel_bandwidth_=
    fx_param_double_req(fx_root,"svm_kernel_bandwidth");

  for(index_t i=0;i<smoothing_kernel_bandwidth_vector_length;i++){
    
    smoothing_kernel_bandwidth_cv_=
      smoothing_kernel_bandwidth_vector[i];
    
    for(index_t j=0;j<lambda_vector_length;j++){
      
      lambda_cv_=lambda_vector[j];
	
      for(index_t fold_num=0;fold_num<k_folds_;fold_num++){
	
	printf("i=%d,j=%d,fold_num=%d..\n",i,j,fold_num);
	Dataset cv_train_data,cv_test_data;
	Vector cv_train_labels,cv_test_labels;
	
	// Will get the train and test folds 
	//printf("Get the fold...\n");
	GetTheFold_(cv_train_data,cv_test_data,cv_train_labels, 
		    cv_test_labels,fold_num);

	//printf("cv train labels is...\n");
	//cv_train_labels.PrintDebug();
	
	//printf("cv test labels is...\n");
	//cv_test_labels.PrintDebug();
	
	// This routine will take the train fold and the test fold
	// and solve the local SVM problem

	RunLocalKernelMachines_(cv_train_data.matrix(),cv_test_data.matrix(),
				cv_train_labels);
      }
    }
  }
}

 
template <typename TKernel> void LocalKernelMachines<TKernel>::PerformCrossValidation_(){

  
  CrossValidateOverSmoothingKernelBandwidthAndLambda_();
  
  // We now have the optimal parameters.
}
 
template <typename TKernel> void LocalKernelMachines<TKernel>::PrepareForCrossValidation_(){
  
  // The purpose of this function is simply to setup the stage for crossvalidation. 
  
  // If no crossvalidation is required then this code will not be invoked.,
  
  // In this case we are crossvalidating. hence lets stich the 
  // train data with the train labels.
  
  Matrix stiched_train_data;
  printf("Train labels are....\n");
  train_labels_vector_.PrintDebug();
  AppendVectorToMatrixAsLastRow(train_labels_vector_,train_data_,
				stiched_train_data);
  
  void *random_permutation;
  random_permutation=(void *)malloc(num_train_points_*sizeof(index_t));
  math::MakeRandomPermutation(num_train_points_,(index_t *) random_permutation);
  
  // Lets typecast random_permutation as ArrayList <index_t>
  
  random_permutation_array_list_.InitAlias((int *)random_permutation,num_train_points_);
  
   
  // Now set the matrix part of dset to the train dataset
  
  dset_.matrix().Copy(stiched_train_data);
  
  printf("training data is...\n");
  dset_.matrix().PrintDebug();
}
   
template <typename TKernel> void LocalKernelMachines< TKernel>::SetUpCrossValidationFlags_(){
    

  printf("Setting up corssvalidation flags......\n");
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
    
    printf("We dont need to crossvalidate...\n");
    // Straight away run the local kernel machine with the given parameters.
    
    // We don't use dset here. Hence set it to empty so as to avoid segfault
    
    dset_.InitBlank();
    
  }
}
#endif
