#ifndef LOCAL_KERNEL_MACHINES_IMPL_H_
#define LOCAL_KERNEL_MACHINES_IMPL_H_
#include "ocas.h"
#include "ocas_smo.h"
#include "ocas_line_search.h"
#include "range_search.h"


template <typename TKernel> double  LocalKernelMachines< TKernel>:: 
RunLocalKernelMachines_(Matrix &train_data_local, 
			Matrix &test_data_local, 
			Vector &train_labels_local,
			Vector &test_labels_local,
			double smoothing_kernel_bandwidth,
			double lambda, 
			ArrayList< ArrayList <double> > &
			smoothing_kernel_values_in_range,
			ArrayList< ArrayList <int> > &indices_in_range){
  
  int num_mistakes=0;
  
  
  // We first start by finding for each test point, points that lie in
  // its sigma neighbourhood.
  
  // get the indices in range and their corresponding kernel values
/*   ArrayList< ArrayList <int> > indices_in_range; */
/*   ArrayList< ArrayList <double> > smoothing_kernel_values_in_range; */

/*   RangeSearch rs; */
/*   rs.Init(train_data_local,test_data_local,smoothing_kernel_bandwidth); */
/*   rs.PerformRangeSearch(); */
/*   rs.get_indices_in_range(indices_in_range); */
/*   rs.get_smoothing_kernel_values_in_range(smoothing_kernel_values_in_range); */

  //printf("Range search completed...\n");
  
  for(index_t i=0;i<test_data_local.n_cols();i++){
     
    // The label local svm will predict
    
    double y_pred_label;
    if(smoothing_kernel_values_in_range[i].size()==0){

      // This means there are no points in this neighbourhood. 
      // Hence assign a random label.


      double rand_num=math::Random(-1,1);
      y_pred_label=rand_num>0?1.0:-1.0;
      //y_pred_label=2.0;
      //printf("Empty neighbourhood...\n");
    }
    else{
   
      // Perform Optimization.
      OCAS ocas;
      ocas.Init(test_data_local.GetColumnPtr(i),train_data_local,
		train_labels_local,indices_in_range[i],
		smoothing_kernel_values_in_range[i],smoothing_kernel_bandwidth,
		lambda);
      ocas.Optimize(); 
      
      Vector w_opt;
      w_opt.Init(num_dims_);
      ocas.get_optimal_vector(w_opt);
      double w_opt_at_x=
	la::Dot(num_dims_,w_opt.ptr(),test_data_local.GetColumnPtr(i));

      if(w_opt_at_x>0){
	y_pred_label=1.0;
      }
      else{
	
	y_pred_label=-1.0;
      }
      // printf("Performed optimization...\n");
    }
    if(fabs(test_labels_local[i]-y_pred_label)>SMALL){

      // printf("true test label=%f...\n",test_labels_local[i]);
      //printf("Predicted label=%f...\n",y_pred_label);
      //printf("Made a mistake...\n");
      num_mistakes++;
    } 

    //printf("Finished working on point:%d..\n",i);
  }
  //  printf("Finished working on this data...\n");
  // Return the error rate

  return (double)num_mistakes/test_data_local.n_cols();
}

template <typename TKernel> void LocalKernelMachines< TKernel>:: 
FindPointsInNeighbourhood_( Matrix &ref_data, double *q_point,  
			    ArrayList <int> &indices_in_range,  
			    ArrayList<double> &smoothing_kernel_values_in_range, 
			    double smoothing_kernel_bandwidth){ 
  
  
  
  Vector q_point_vec; 
  q_point_vec.Alias(q_point,num_dims_); 
  //printf("Query point is...\n"); 
  //q_point_vec.PrintDebug(); 
  
  
  EpanKernel ek; 
  ek.Init(smoothing_kernel_bandwidth); 
  index_t length=0; 
  
  for(index_t i=0;i<ref_data.n_cols();i++){ 
    
    double sqd_distance= 
      la::DistanceSqEuclidean(num_dims_,ref_data.GetColumnPtr(i),q_point); 
    
    
    double distance=sqrt(sqd_distance); 
    
    double kernel_value=ek.EvalUnnorm(distance); 
    //double norm_const=ek.CalcNormConstant(num_dims_); 
    
    double norm_const=1.0; 
    kernel_value/=norm_const; 
    
  
    if(distance<smoothing_kernel_bandwidth){ 
      
      //This point is in the range
      if(length<20){ 
	
 	//We have enough space 
 	indices_in_range[length]=i; 
 	smoothing_kernel_values_in_range[length]=kernel_value;
      } 
      else{ 
 	// printf("Adding extra space....\n"); 
 	// We dont have enough space. Hence allocate space for one more element 
 	indices_in_range.PushBack(1); 
 	smoothing_kernel_values_in_range.PushBack(1); 
 	indices_in_range[length]=i; 
 	smoothing_kernel_values_in_range[length]=kernel_value; 
      }  
       length++;
    } 
  } 
  //printf("Length is %d...\n",length); 
  // If we have less than 20 elements then lets remove the extra space */
  
  if(length<20){ 
    indices_in_range.Remove(length,20 -length); 
    smoothing_kernel_values_in_range.Remove(length,20 -length); 
  } 
} 
  

template <typename TKernel> void LocalKernelMachines< TKernel>::TrainLocalKernelMachines(){


  // Even before we train, lets append the training and test datasets
  // with a vector of 1's as the additional dimension

  Vector ones;

  ones.Init(num_train_points_);
  
  ones.SetAll(1);
  
  // Now append this set of 1's to the train data

  Matrix train_data_appended;
  Matrix test_data_appended;

  AppendVectorToMatrixAsLastRow(ones,train_data_,train_data_appended);  
  AppendVectorToMatrixAsLastRow(ones,test_data_,test_data_appended);

  num_dims_+=1;


  // Get the cv flag. By default we will crossvalidate
  int cv_flag=
    fx_param_int(fx_root,"cv_flag",1);

  if(cv_flag==1){

    printf("I am performing crossvalidation....\n");

    // Get the number of folds of crossvalidation
    
    k_folds_=fx_param_int(fx_root,"k_folds",5);
    
    // In this case do crossvalidation
    CrossValidation_(train_data_appended,test_data_appended);
    
    // Having performed crossvalidation (if it was required) 
  // we now have the necessary parameters
    
    printf("Run the code with the optimal parameter setting...\n");

    ArrayList< ArrayList <int> > indices_in_range;
    ArrayList< ArrayList <double> > smoothing_kernel_values_in_range;
    
    RangeSearch rs;
    rs.Init(train_data_appended,test_data_appended,
	    optimal_smoothing_kernel_bandwidth_);
    rs.PerformRangeSearch();
    rs.get_indices_in_range(indices_in_range);
    rs.get_smoothing_kernel_values_in_range(smoothing_kernel_values_in_range);

    double error_rate=
      RunLocalKernelMachines_(train_data_appended,test_data_appended,
			      train_labels_vector_,test_labels_vector_,
			      optimal_smoothing_kernel_bandwidth_,
			      optimal_lambda_,
			      smoothing_kernel_values_in_range,
			      indices_in_range);
    printf("Final error rate is %f..\n",error_rate);
    printf("optimal_smoothing_kernel_bandwidth=%f...\n",
	   optimal_smoothing_kernel_bandwidth_);
    printf("optimal_lambda=%f...\n",
	   optimal_lambda_);
  }
  else{
    
    
    optimal_smoothing_kernel_bandwidth_=
      fx_param_double(fx_root,"smoothing_kernel_bandwidth",-DBL_MAX);
    
    
    //Get the regularization parameter
    
    optimal_lambda_= fx_param_double(fx_root,"lambda",-DBL_MAX);
    
    printf("Will run local linear machine with the following parameters...\n");
    printf("lambda=%f..\n",optimal_lambda_);
    printf("Optimal smoothing kernel bandwidth=%f",
	  optimal_smoothing_kernel_bandwidth_);


    ArrayList< ArrayList <int> > indices_in_range;
    ArrayList< ArrayList <double> > smoothing_kernel_values_in_range;
    
    RangeSearch rs;
    rs.Init(train_data_appended,test_data_appended,
	    optimal_smoothing_kernel_bandwidth_);
    rs.PerformRangeSearch();
    rs.get_indices_in_range(indices_in_range);
    rs.get_smoothing_kernel_values_in_range(smoothing_kernel_values_in_range);
    
    double error_rate=
      RunLocalKernelMachines_(train_data_appended,test_data_appended,
			      train_labels_vector_,test_labels_vector_,
			      optimal_smoothing_kernel_bandwidth_,
			      optimal_lambda_,
			      smoothing_kernel_values_in_range,
			      indices_in_range);
    
    printf("final error rate =%f..\n",error_rate);
    
  }
}
  
template <typename TKernel> void LocalKernelMachines< TKernel>::
Init(Matrix &train_data,Matrix &test_data,Vector &train_labels_vector,
     Vector &test_labels_vector,
     struct datanode *module_in){
    
  //Initialize
    
  module_=module_in;
  train_data_.Alias(train_data);
  test_data_.Alias(test_data);
    
  //printf("Train labels vector in main function is...\n");
  //train_labels_vector.PrintDebug();
  train_labels_vector_.Alias(train_labels_vector);
  test_labels_vector_.Alias(test_labels_vector);  
    
  num_train_points_=train_data_.n_cols();
  num_test_points_=test_data_.n_cols();
  num_dims_=train_data_.n_rows();

  

  // Find out the kernels being used

  const char *smoothing_kernel=
    fx_param_str(fx_root,"smoothing_kernel","epan");
    

  if(strcmp(smoothing_kernel,"epan")==0){
      
    smoothing_kernel_=EPAN_SMOOTHING_KERNEL;
      
  }
  else{
      
    //smoothing_kernel_=UNIFORM_SMOOTHING_KERNEL;
  }
  //printf("Basic initializations done...\n");
}
#endif
