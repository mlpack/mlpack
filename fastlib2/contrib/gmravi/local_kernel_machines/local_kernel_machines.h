

#ifndef LOCAL_KERNEL_MACHINES_H
#define LOCAL_KERNEL_MACHINES_H

template<typename TKernel>
class LocalKernelMachines{

  FORBID_ACCIDENTAL_COPIES(LocalKernelMachines)
 private:
  

  //The essentials
  Matrix train_data_;

  Matrix test_data_;

  index_t num_train_points_;

  index_t num_test_points_;

  index_t num_dims_;

  // The regularization constant

  double reg_constant_;

  //Bandwidth of the similarity kernel (if used);

  double optimal_similarity_kernel_bandwidth_;

  //Optimal bandwidth of the smoothing kernel used
  double optimal_smoothing_bandwidth_;
  


  
 public:


  void PerformCrossValidation_(){
    
    //Lets do a 5-fold crossvalidation



  }

  void TrainLocalKernelMachines(){
    
    //Check if cross-validation is required

    cv_flag=fx_param_int_req(module_,"cv_flag");

    if(cv_flag==1){
      
      // We will do crossvalidation
      PerformCrossValidation_();
      
      
    }
    else{
      
      // We are magically give bandwidths and hence we will solve all 
      // Optimization problems with this bandwidth

      optimal_similarity_bandwidth_=
	fx_param_int_req(module_,"similarity_bandwidth");

      optimal_smoothing_bandwidth_=
	fx_param_int_req(module_,"smoothing_bandwidth");
    }
      
  }
  
  void Init(Matrix &train_data,Matrix &test_data,struct datanode *module_in)
    
    //Initialize
    
    module_=module_in;
    train_data.Alias(train_data);
    test_data_.Alias(test_data);
    
    num_train_points_=train_data_.n_cols();
    num_test_points_=test_data_.n_cols();
    num_dims_=train_data_.n_rows();
    
    
    
}

#endif
