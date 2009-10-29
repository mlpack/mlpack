#include "local_kernel_machines.h"
#include ""

int main(int argc, char *argv[]){

  // Initialize FastExec

  fx_init(argc, argv, NULL);

  // Do the usual initialization.

  struct datanode *lkm_module=fx_submodule(fx_root,"lkm");

  LocalKernelMachines lkm;

  
  const char *train_data_file_name=fx_param_str_req(fx_root,"train");

  Matrix train_data;
  data::Load(train_data_file_name,&train_data);


  const char *test_data_file_name=fx_param_str_req(fx_root,"test");

  Matrix test_data;
  data::Load(test_data_file_name,&test_data);

  //Initialize 

  lkm.Init(train_data,test_data,lkm);

  // Train

  lkm.TrainLocalKernelMachines();



}
