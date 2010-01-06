#include "fastlib/fastlib.h"
#include "mlpack/svm/svm.h"
#include "local_kernel_machines_def.h"
#include "local_kernel_machines_impl.h"
#include "utils.h"
#include "my_crossvalidation.h"
int main(int argc, char *argv[]){

  // Initialize FastExec

  fx_module *root = fx_init(argc, argv, NULL);

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
  
   // test labels
  const char *test_labels_file_name=
    fx_param_str_req(fx_root,"test_labels");  
  

  Matrix train_data;
  data::Load(train_data_file_name,&train_data);

  // Input parameter is test data
  Matrix test_data;
  data::Load(test_data_file_name,&test_data);

  // The train labels
  
  Matrix train_labels_mat;
  data::Load(train_labels_file_name,&train_labels_mat);

  Matrix test_labels_mat;
  data::Load(test_labels_file_name,&test_labels_mat);
 
  // Convert the above matrix to a vector

  Vector train_labels;
  ConvertOneColumnMatrixToVector(train_labels_mat,train_labels);
  
  Vector test_labels;
  ConvertOneColumnMatrixToVector(test_labels_mat,test_labels);

  // Get the kernels used

  const char *smoothing_kernel=
    fx_param_str(fx_root,"smoothing_kernel","epan");
  
  const char *similarity_kernel=
    fx_param_str(fx_root,"svm_kernel","linear");
  
  
  if (strcmp(similarity_kernel,"linear")==0){
    
    // The similarity kernel being used is the Linear kernel
    LocalKernelMachines <SVMLinearKernel> lkm;
    lkm.Init(train_data,test_data,train_labels,test_labels,lkm_module);

    // Train
    fx_timer_start(NULL,"train");
    lkm.TrainLocalKernelMachines();
    printf("Done with all calculations...\n");
    fx_timer_stop(NULL,"train");
  
  }
  else{
    
    printf("We dont yet support SVMRBF kernels..\n");
  
  }
  fx_done(NULL);


}
