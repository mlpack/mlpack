#include "local_kernel_machines_def.h"
#include "utils_lkm.h"

#include "fastlib/fastlib.h"
#include "mlpack/svm/svm.h"


int main(int argc, char *argv[]){

  // Initialize FastExec

  fx_init(argc, argv, NULL);

  // Do the usual initialization.

  struct datanode *lkm_module=fx_submodule(fx_root,"lkm");


  // train is the input training data
  const char *train_data_file_name=
    fx_param_str_req(fx_root,"train");

  // test is the input test data
  const char *test_data_file_name=
    fx_param_str_req(fx_root,"test");  

  // train labels
  const char *train_labels_file_name=
    fx_param_str_req(fx_root,"train_labels");  
  
  
  

  Matrix train_data;
  data::Load(train_data_file_name,&train_data);

  // Input parameter is test data
  Matrix test_data;
  data::Load(test_data_file_name,&test_data);

  // The train labels
  
  Matrix train_labels_mat;
  data::Load(train_labels_file_name,&train_labels_mat);

 
  // Convert the above matrix to a vector

  Vector train_labels;
  ConvertMatrixToVector(train_labels_mat,train_labels);
  

  // Get the kernels used

  const char *smoothing_kernel=
    fx_param_str(fx_root,"smoothing_kernel","epan");

  const char *similarity_kernel=
    fx_param_str(fx_root,"svm_kernel","linear");
  
  if (strcmp(similarity_kernel,"linear")){
    
    // The similarity kernel being used is the Linear kernel
    LocalKernelMachines <SVMLinearKernel> lkm;
    lkm.Init(train_data,test_data,train_labels,lkm_module);

    // Train
    lkm.TrainLocalKernelMachines();
  
  }
  else{
    
    //The similarity kernel being used is the SVMRBF kernel
    LocalKernelMachines<SVMRBFKernel> lkm;
    lkm.Init(train_data,test_data,train_labels,lkm_module);

    // Train
    lkm.TrainLocalKernelMachines();
  
  }
}
