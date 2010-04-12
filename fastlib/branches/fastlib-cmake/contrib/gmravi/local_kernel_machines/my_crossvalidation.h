#include "utils.h"
#ifndef MY_CROSSVALIDATION_H
#define MY_CROSSVALIDATION_H


/////////////////// Set up the parameters for Crossvalidation//////////////////////

template <typename TKernel> void  LocalKernelMachines <TKernel>::
GenerateSmoothingBandwidthVectorForCV_(Vector &smoothing_bandwidth_vector){
  
  int num_smoothing_bandwidth=8;
   smoothing_bandwidth_vector.Init(num_smoothing_bandwidth);

   double low=0.005;
   double hi=0.8;
   double gap=(hi-low)/num_smoothing_bandwidth;
  for(index_t i=0; i<num_smoothing_bandwidth;i++){
    smoothing_bandwidth_vector[i]=low+gap*i;
  }
  printf("Crossvalidating over smoothing kernel bandwidth....\n");
  smoothing_bandwidth_vector.PrintDebug();
}

template <typename TKernel> void  LocalKernelMachines <TKernel>::
GenerateLambdaVectorForCV_(Vector &lambda_vector){
  
  int num_lambda=8;
  double low=pow(10,-4);
  lambda_vector.Init(num_lambda);
  for(index_t i=0; i<num_lambda;i++){

     lambda_vector[i]=0.5+0.5*i;
    //lambda_vector[i]=low*pow(10,i);
  }
  printf("lambda vector for crossvalidation is ...\n");
  lambda_vector.PrintDebug();
}

/////////////////////////////////////////////////

 
template <typename TKernel> void  LocalKernelMachines <TKernel>:: 
GetTheFold_(Dataset &cv_train_data_appended,Dataset &cv_test_data_appended,
	    Vector &cv_train_labels, Vector &cv_test_labels,index_t fold_num)
{
  
  // The crossvalidation folds
  
  dset_.SplitTrainTest(k_folds_,fold_num,random_permutation_array_list_,
		       &cv_train_data_appended,&cv_test_data_appended);
  
  // In the stiched dataset, the last column was the label column. .
  // So now remove the label column.
  
  RemoveLastRowFromMatrixInit(cv_train_data_appended.matrix(),cv_train_labels);
  RemoveLastRowFromMatrixInit(cv_test_data_appended.matrix(),cv_test_labels);
}

template <typename TKernel> void LocalKernelMachines<TKernel>::
CrossValidateOverSmoothingKernelBandwidthAndLambda_(){
  
  
  Vector smoothing_kernel_bandwidth_vector;
  Vector lambda_vector;
  
  GenerateSmoothingBandwidthVectorForCV_(smoothing_kernel_bandwidth_vector);
  GenerateLambdaVectorForCV_(lambda_vector);
  
  index_t smoothing_kernel_bandwidth_vector_length=
    smoothing_kernel_bandwidth_vector.length();
  
  index_t lambda_vector_length=
    lambda_vector.length();
  
   double optimal_error_rate=1.0;
  
  for(index_t i=0;i<smoothing_kernel_bandwidth_vector_length;i++){
    
    double smoothing_kernel_bandwidth=
      smoothing_kernel_bandwidth_vector[i];
    
    for(index_t j=0;j<lambda_vector_length;j++){
      
      double lambda=lambda_vector[j];

      double average_error_rate=0.0;
	
      for(index_t fold_num=0;fold_num<k_folds_;fold_num++){
	

	Dataset cv_train_data_appended,cv_test_data_appended;
	Vector cv_train_labels,cv_test_labels;
	
	// Will get the train and test folds 
	//printf("Get the fold...\n");

	GetTheFold_(cv_train_data_appended,cv_test_data_appended,
		    cv_train_labels,cv_test_labels,fold_num);
	
	
	// This routine will take the train fold and the test fold
	// and solve the local SVM problem

        average_error_rate+=
	  RunLocalKernelMachines_(cv_train_data_appended.matrix(),
				  cv_test_data_appended.matrix(),
				  cv_train_labels,cv_test_labels,
				  smoothing_kernel_bandwidth,lambda);
      }
      average_error_rate/=k_folds_;
      if(average_error_rate<optimal_error_rate){
	
	optimal_error_rate=average_error_rate;
	optimal_smoothing_kernel_bandwidth_=smoothing_kernel_bandwidth;
	optimal_lambda_=lambda;
	
      }

      printf("bandwidth=%f, lambda=%f, error_rate=%f...\n",
	     smoothing_kernel_bandwidth_vector[i],
	     lambda_vector[j],average_error_rate);
     
    }
  }

  printf("Optimal cv error rate=%f...\n",optimal_error_rate);
  printf("optimal_lambda_=%f...\n",optimal_lambda_);
  printf("Optimal bandwidth=%f...\n",optimal_smoothing_kernel_bandwidth_);
}

 
template <typename TKernel> void LocalKernelMachines<TKernel>::
PrepareForCrossValidation_(Matrix &train_data_appended, Matrix &test_data_appended){
  
  // The purpose of this function is simply to setup the stage for
  // crossvalidation.
  
  // If no crossvalidation is required then this code will not be
  // invoked.,
  
  // In this case we are crossvalidating. hence lets stich the train
  // data with the train labels.
  
  Matrix stiched_train_data_appended;

  AppendVectorToMatrixAsLastRow(train_labels_vector_,train_data_appended,
				stiched_train_data_appended);
  
  void *random_permutation;
  random_permutation=(void *)malloc(num_train_points_*sizeof(index_t));
  math::MakeRandomPermutation(num_train_points_,
			      (index_t *) random_permutation);
  
  // Lets typecast random_permutation as ArrayList <index_t>
  
  random_permutation_array_list_.Copy((int *)random_permutation,
					   num_train_points_);
   
  // Now set the matrix part of dset to the train dataset
  
  dset_.matrix().Copy(stiched_train_data_appended);

  free(random_permutation);
}
   

template <typename TKernel> void LocalKernelMachines<TKernel>::
CrossValidation_(Matrix &train_data_appended,
		 Matrix &test_data_appended){
  
  
  PrepareForCrossValidation_(train_data_appended,test_data_appended);
  CrossValidateOverSmoothingKernelBandwidthAndLambda_();
}

#endif
