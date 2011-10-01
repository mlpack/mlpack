#include "hk_convex_ise.h"
#include "fastlib/fastlib_int.h"
#include "/net/hc295/gmravi/home/fastlib/fastlib2_int/fastlib2/mlpack/kde"
int main(int argc, char *argv[]){
  
  //Initalize FastExec
  fx_init(argc,argv,NULL);
  
  struct datanode* ise_module=fx_submodule(NULL,"ise");
  
  //Train data file is a reuqirement
  const char *train_file=fx_param_str_req(ise_module,"train");
  
  Matrix train_data;

  Matrix test_data;

  Vector hyperkernel_test_densities;

  
  //Load the train and the test datasets
  
  data::Load(train_file,&train_data);

  if(fx_param_exists(module_,"test")){
    
    //Load the dataset
    const char *test_file=fx_param_str_req(module_,"test");
    data::Load(test_file,&test_data_);
    //num_test_points_=test_set_.n_cols();
    
    //Since there is a  test file hence 
    //We shall compute the test densities
    computed_test_densities_.Init(test_data_.n_cols());
  }
  else{
    
    //This avoids segmentation fault
    test_data_.Init(0,0); 
    computed_test_densities_.Init(0);
    //num_test_points_=0;
  }
  
  //Having got all the initial data lets create an object of the class HkIse
  
  HkConvexIse hk_convex_ise;
  
  //Initialize the object and call functions on it
  
  hk_convex_ise.Init(train_data,test_data,ise_module);
  hk_convex_ise.ComputeOptimalKernel();

  if(fx_param_exists(NULL,"test")){

    hk_convex_ise.GetTestDensities(); 
  }
  

  

